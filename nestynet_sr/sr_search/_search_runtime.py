# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""The closure-heavy Stage-A separability driver, preserved as one state machine."""

import copy
import math
from typing import Any, Dict, Iterable, List, Optional, Tuple
import torch
from torch.utils.data import DataLoader, TensorDataset
from nestynet_sr.sr_core import Var, ast_to_human_readable, build_monomial_ast, check_separability_ops, collect_nn_atoms, replace_atom_in_ast
from nestynet_sr.sr_core.bridges import AtomNode, ConstNode, CosNode, MulNode, SinNode, _collect_var_idxs_from_node, effective_arity, eval_inputs, get_input_exprs, has_nontrivial_input, is_trivial_input
from nestynet_sr.sr_core.fit_links import canonical_fit_link_name, describe_fit_link
from .ast_utils import compact_expression_repr as _compact_expression_repr
from .features import TrigAxisSpec, discover_constant_directions, discover_invariance_features, discover_model_directions, discover_parity_axes, discover_poly_in_f2, discover_poly_in_x, discover_preferred_origins, discover_radial_groups, discover_rational_poly, discover_saturating_axes, discover_scaling_features, discover_trig_axes, poisson_profile, probe_oracle_scaling, probe_trig_scaling, sample_line_curvature, trig_from_profile, verify_compound_null_test
from .model_builders import build_composite_ast, is_minimal_ast
from .model_selection import apply_noise_floor_to_acceptance_thresholds as _apply_noise_floor_to_acceptance_thresholds, compute_accept_threshold as _compute_accept_threshold, estimate_transform_noise_floor_raw as _estimate_transform_noise_floor_raw, loss_excess_above_floor as _loss_excess_above_floor, resolve_acceptance_noise_floor_raw as _resolve_acceptance_noise_floor_raw
from .training import train_candidate_model, train_initial_model
from .stagea_fit_tournament import (
    fit_initial_model_with_tournament,
    fit_stageA_candidate_with_tournament,
)
from .wrapper_policy import snap_omega
from .y_transforms import precision_for_transform

from ._search_shadow import (
    BLUE,
    GREEN,
    RED,
    RESET,
    YELLOW,
    _aggregate_losses,
    _analytic_units_rejection,
    _apply_fit_link_to_model,
    _ast_matches_arch,
    _clone_reuse_leaves,
    _loss_str,
    _sanitize_func_tensors,
    _should_skip_compound_extension_after_sep,
    _stageA_reset_shadow_registry,
    _stageA_shadow_registry,
    _stageA_sync_shadow_registry,
)
from ._search_proposals import (
    _COMPOUND_Z_TOKEN,
    _atom_compound_cols,
    _check_separability_in_input_space,
    _compound_best_proposal_confidence,
    _is_pure_1d_full_compound_ast,
    _phase_hint_compound_proposals_for_atom,
    _quick_separability_check,
    _separability_proposal_to_ast_unified,
    _stageA_append_compound_replay_proposals,
    _stageA_ast_fingerprint,
    _stageA_noisy_soft_monomial_product_proposals_from_scaling,
    _stageA_split_group_record_payload,
    _try_stageA_univariate_monomial_for_atom,
)
from ._search_policy import (
    _accept_threshold_with_structural_target,
    _candidate_metric,
    _format_stageA_overlap_split_committee_report,
    _is_clean_disjoint_cover,
    _is_singleton_disjoint_cover,
    _nn_split_signature,
    _sep_metric_to_score,
    _stageA_best_disjoint_separability_metric,
    _stageA_destructive_prune_committee_gate,
    _stageA_leaf_projection_nonregression_override,
    _stageA_leaf_prune_acceptance_gate,
    _stageA_loss_budget_multiplier,
    _stageA_noisy_overlap_split_gate,
    _stageA_overlap_split_committee_gate,
    _stageA_provisional_full_refit_failure_status,
    _stageA_provisional_move_reason,
    _stageA_under_protest_threshold_cap,
)
from ._search_training import (
    _axis_is_coupled_by_invariance,
    _axis_is_inside_compound_input,
    _build_additive_gauge_fix_factories,
    _build_leaf_prune_candidate_ast,
    _build_multiplicative_gauge_fix_factories,
    _build_tag_to_leaf_map,
    _build_xtransformed_loaders,
    _check_asinh_yspace_sanity,
    _compute_atom_scale,
    _compute_y_med_mad_from_loader,
    _detect_leaf_nondep_axes_for_atom,
    _eval_ast_subtree_on_data,
    _evaluate_overlap_truth_metric,
    _extract_compound_targets_from_ast,
    _find_residual_refit_context,
    _loader_all_finite,
    _nn_multivar_axes,
    _overlap_gauge_stage_is_feasible,
    _overlap_truth_metric_is_acceptable,
    _propose_reciprocal_x_map,
    _snap_omega,
    _stageA_identity_target_good,
    _stageA_initial_fit_restart_allowed,
    _teacher_init_additive,
    _teacher_init_multiplicative,
    _try_asinh_fit,
)
from ._search_detection import (
    _check_early_compound_from_scaling,
    _detect_compound_variable_for_atom,
    _detect_pure_difference_compounds,
    _get_qualifying_scaling_vars,
    _try_early_compound_candidate,
)
from ._search_structure import (
    _loader_n_eff,
    _stageA_append_visible_buckingham_1d_prefactor_proposals,
    _try_stageA_additive_shared_response_block,
)
from ._search_compounds import (
    _try_compound_candidates_for_atom,
    _try_stageA_compound_during_sep_for_atom,
    _try_stageA_decisive_gs_preflight_for_atom,
    _try_stageA_terminal_closure_probe,
)


def _stageA_record_decisive_gs_preflight_best_effort(
    *,
    candidate_model,
    parent_ast,
    candidate_ast,
    parent_loss,
    candidate_loss,
    full_compound: bool,
    search_hp,
    move_details,
    record_move,
    sync_shadow,
) -> None:
    """Record an already-committed GS preflight without reopening fallback.

    Candidate acceptance is the transaction boundary.  Diagnostics that fail
    after that boundary must not make the caller treat the accepted candidate
    as a failed preflight and run legacy proposers on partially updated state.
    """

    try:
        details = move_details(candidate_model, bool(full_compound))
    except Exception as exc:
        details = {"full_compound": bool(full_compound)}
        print(
            "[Stage A GS Preflight] Could not collect move details after "
            f"acceptance: {type(exc).__name__}: {exc}"
        )
    try:
        record_move(
            move_kind="decisive_gs_preflight",
            parent_ast=parent_ast,
            candidate_ast=candidate_ast,
            parent_loss=parent_loss,
            candidate_loss=candidate_loss,
            reason=(
                "certified full-support GS carrier accepted before legacy "
                "early compounds"
            ),
            risk_tags={"compound_coordinate"},
            details=details,
        )
    except Exception as exc:
        print(
            "[Stage A GS Preflight] Move recording failed after acceptance: "
            f"{type(exc).__name__}: {exc}"
        )
    try:
        sync_shadow(
            search_hp,
            candidate_ast,
            reason="decisive GS preflight",
        )
    except Exception as exc:
        print(
            "[Stage A GS Preflight] Shadow-registry sync failed after "
            f"acceptance: {type(exc).__name__}: {exc}"
        )


def run_separability_for_transform(
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
    stageA_restart_callback=None,
):
    """
    Run the greedy separability search for a single y-transform.

    Parameters
    ----------
    i_op              : int
        Index in the y-transform list.
    y_op, y_op_inv    : callables | None
        Forward and inverse operators on y.
    candidate_sep_ops : list[bool]
        Current flags for which y_ops are worth trying.
    initial_ast       : Node
        Initial AST to start from.
    filepath          : str
    Nxvars            : int
    y_med, y_mad      : floats
    np_dtype, dtype   : numpy dtype, torch dtype
    device            : torch.device
    data_hp, model_hp, lm_hp, search_hp : config dataclasses
    leaf_builder      : LeafBuilder
    model_output      : str
    model_sep_output  : str
    mode              : str
        "quick", "full", or "fit_only" to indicate which pass this is.
        "fit_only" returns after initial fit without separability search.
    y_raw_full        : array-like | None
        Full raw-y dataset used for transform-aware noise-floor estimation.
    noise_sigma_y     : float | None
        Known homoscedastic additive y-noise standard deviation in raw units.
    noise_floor_mc_samples : int
        Number of Monte-Carlo perturbations used for the noise-floor estimate.

    Returns
    -------
    separability_success : bool
    model                : CompositeAdaptor | None
    rest_add             : list[int] | None
    rest_mult            : list[int] | None
    candidate_sep_ops    : updated list[bool]
    current_ast          : Node | None
    last_resort_suggested: bool
        Deprecated compatibility flag (always False in unified y-search flow).
    full_compound_solved : bool
        True if a compound covering ALL input variables achieved target loss.
        Caller should skip remaining y-transforms since f(z) fully explains data.
    """
    # Optional: aggregate units-based rejections for a compact end-of-run summary.
    from collections import Counter, defaultdict

    from .data_utils import build_datasets, build_datasets_multi  # avoid circular import

    _units_reject_counts = defaultdict(Counter)
    stageA_move_records: List[Dict[str, Any]] = []
    stageA_provisional_commits: List[Dict[str, Any]] = []
    stageA_rejected_transactions: set = set()
    stageA_rejection_records: List[Dict[str, Any]] = []
    pending_stageA_full_refit_transaction: Optional[Dict[str, Any]] = None
    stageA_move_seq = 0

    # Predefine so summary helpers are safe even on early returns.
    y_transform_name = None
    units_spec = None
    units_raw_x_dims = None

    def _units_reason_key(reason):
        try:
            s = str(reason)
        except Exception:
            s = repr(reason)
        s = (s or "").strip()
        if not s:
            s = "<no reason>"
        s = s.splitlines()[0].strip()
        if len(s) > 160:
            s = s[:157] + "..."
        return s

    def _units_reject(kind: str, reason):
        if not (bool(enforce_units) and (units_spec is not None)):
            return
        _units_reject_counts[str(kind)][_units_reason_key(reason)] += 1

    def _units_print_summary():
        if not (bool(enforce_units) and (units_spec is not None)):
            return
        total = sum(sum(c.values()) for c in _units_reject_counts.values())
        if total <= 0:
            return
        try:
            yname = y_transform_name or y_op_str
        except Exception:
            yname = y_op_str

        label_map = {
            "compound_variant": "compound variants",
            "separability_variant": "separability variants",
            "x_transform_candidate": "x-transform candidates",
            "x_precond_reciprocal": "x-preconditioning (reciprocal)",
            "x_precond_trig": "x-preconditioning (trig)",
            "early_compound_trig": "Early Compound trig extensions",
        }

        print(f"[Units] Stage A summary ({yname}): skipped {total} unit-incompatible proposal(s).")
        for kind, counter in sorted(
            _units_reject_counts.items(), key=lambda kv: -sum(kv[1].values())
        ):
            k_total = sum(counter.values())
            if k_total <= 0:
                continue
            label = label_map.get(kind, kind)
            print(f"[Units]   - {label}: {k_total}")
            for reason, cnt in counter.most_common(5):
                print(f"[Units]       {cnt}× {reason}")
            if len(counter) > 5:
                other = k_total - sum(cnt for _, cnt in counter.most_common(5))
                if other > 0:
                    print(f"[Units]       (+{other} other reason(s))")

    deferred_stageA_branches: List[Dict[str, Any]] = []

    def _attach_stageA_move_records(target_model):
        if target_model is None:
            return
        try:
            setattr(target_model, "_stageA_move_records", [dict(r) for r in stageA_move_records])
        except Exception:
            pass

    def _attach_stageA_provisional_commits(target_model):
        if target_model is None:
            return
        try:
            setattr(
                target_model,
                "_stageA_provisional_commits",
                [dict(r) for r in stageA_provisional_commits],
            )
        except Exception:
            pass

    def _attach_stageA_rejection_records(target_model):
        if target_model is None:
            return
        try:
            setattr(
                target_model,
                "_stageA_rejection_records",
                [dict(r) for r in stageA_rejection_records],
            )
        except Exception:
            pass

    def _attach_deferred_stageA_branches(target_model):
        if target_model is None or not deferred_stageA_branches:
            return
        try:
            # Keep this metadata lightweight; the active branch owns the actual
            # model state.  Deferred fit-link branches are advisory rescue
            # candidates, not accepted hidden state.
            setattr(
                target_model,
                "_stageA_deferred_fitlink_branches",
                [dict(branch) for branch in deferred_stageA_branches],
            )
        except Exception:
            pass

    def _units_finalize_return(*vals):
        if len(vals) > 1:
            _attach_deferred_stageA_branches(vals[1])
            _attach_stageA_move_records(vals[1])
            _attach_stageA_provisional_commits(vals[1])
            _attach_stageA_rejection_records(vals[1])
        _units_print_summary()
        return vals

    y_op_str = y_op.__name__ if hasattr(y_op, "__name__") else str(y_op)
    y_op_inv_str = y_op_inv.__name__ if hasattr(y_op_inv, "__name__") else str(y_op_inv)
    _stageA_reset_shadow_registry(search_hp, reason=f"Stage A y-transform {y_op_str}")
    # The preflight ledger prevents retries within this transform only.  A
    # different outer transform has a different teacher and must get its own
    # independently calibrated GS opportunity.
    try:
        search_hp._stageA_decisive_gs_preflight_attempted = set()
    except Exception:
        pass

    verbose_sep = bool(getattr(search_hp, 'verbose_separabilities', False))

    # Work with AST natively throughout
    current_ast = initial_ast

    def cap_num_segments(num_segments, base_segments, model_size_est):
        """
        Reduce num_segments to satisfy model_hp.nparam_max if set.

        Parameters
        ----------
        num_segments  : int
            Proposed number of segments.
        base_segments : int
            Number of segments used to obtain model_size_est.
        model_size_est: int
            Parameter count for base_segments segments.
        """
        if model_hp.nparam_max is None or model_size_est <= 0 or base_segments <= 0:
            return num_segments

        params_per_segment = model_size_est / float(base_segments)
        max_segments_by_param = max(1, int(model_hp.nparam_max // params_per_segment))

        if max_segments_by_param < num_segments:
            est_params = params_per_segment * max_segments_by_param
            msg = "Capping num_segments from {} to {} to respect nparam_max {} (est. params {:.0f})".format(
                num_segments, max_segments_by_param, model_hp.nparam_max, est_params
            )
            if est_params > model_hp.nparam_max:
                msg += " [per-segment {:.0f} params already exceed cap]".format(params_per_segment)
            print(msg)
            num_segments = max_segments_by_param

        return max(num_segments, 1)

    # Quick early-exit: if previous checks indicated no separability for this y_op
    if y_op is not None:
        if candidate_sep_ops[i_op]:
            print("Initial fit suggests separability candidates for y_op {}.".format(y_op_str))
        else:
            print(
                "Initial fit suggests no separability candidates for y_op {}, skipping this operation.".format(
                    y_op_str
                )
            )
            return _units_finalize_return(False, None, None, None, candidate_sep_ops, None, False, False)

    # Best-effort name for this y-transform (used for unit inference).
    y_transform_name = None
    try:
        if y_transform_names is not None and i_op is not None:
            idx = int(i_op)
            if 0 <= idx < len(y_transform_names):
                y_transform_name = str(y_transform_names[idx])
    except Exception:
        y_transform_name = None
    if not y_transform_name:
        y_transform_name = "identity" if (y_op is None) else y_op_str
    try:
        setattr(lm_hp, "coe_current_y_transform_name", str(y_transform_name))
        setattr(search_hp, "coe_current_y_transform_name", str(y_transform_name))
    except Exception:
        pass

    # Build datasets for this y-transform (single-dataset identity path retained)
    is_multi = isinstance(filepath, (list, tuple))
    dataset_ids: Optional[List[str]] = None
    agg_mode: str = "mean"
    agg_weights: Optional[List[float]] = None

    def _loader_size(dl) -> int:
        try:
            ds = getattr(dl, "dataset", None)
            if ds is not None:
                return int(len(ds))
        except Exception:
            pass
        return 0

    if is_multi:
        filepaths = [str(p) for p in filepath]
        dataset_ids = [str(p).split("/")[-1].rsplit(".", 1)[0] for p in filepaths]
        ds_tr_list, ds_va_list, dl_tr_list, dl_va_list = build_datasets_multi(
            filepaths=filepaths,
            Nxvars=Nxvars,
            np_dtype=np_dtype,
            data_hp=data_hp,
            y_op=y_op,
        )
        if dl_tr_list is None or dl_va_list is None:
            return _units_finalize_return(False, None, None, None, candidate_sep_ops, None, False, False)
        # Keep existing variable names as primary dataset aliases (dataset 0)
        dataset_train = ds_tr_list[0]
        dataset_val = ds_va_list[0]
        datagen_train_noshuffle = dl_tr_list[0]
        datagen_val_noshuffle = dl_va_list[0]
        train_loaders_all = list(dl_tr_list)
        val_loaders_all = list(dl_va_list)
        agg_mode = "weighted"
        agg_weights = [float(_loader_size(dl)) for dl in val_loaders_all]
    else:
        dataset_train, dataset_val, datagen_train_noshuffle, datagen_val_noshuffle = build_datasets(
            filepath, Nxvars, np_dtype, data_hp, y_op
        )
        if datagen_train_noshuffle is None:
            return _units_finalize_return(False, None, None, None, candidate_sep_ops, None, False, False)
        train_loaders_all = [datagen_train_noshuffle]
        val_loaders_all = [datagen_val_noshuffle]
        agg_weights = [float(_loader_size(datagen_val_noshuffle))]
        dataset_ids = [str(filepath).split("/")[-1].rsplit(".", 1)[0]] if filepath is not None else ["dataset_0"]

    # Per-dataset model/loss tracking used by multi-dataset Stage-A mode.
    models_multi: Optional[List[torch.nn.Module]] = None
    current_val_losses: Optional[List[float]] = None

    # Track any x-preprocessing substitutions applied during Stage A (e.g., trig feature substitution)
    x_transform_map = {}
    trig_tried_omegas = {}  # axis -> list of omega values already tried for trig preconditioning
    trig_precond_active = None  # (axis, omega) if currently testing a trig transform
    x_precond_saved_state = None  # saved state (x_transform_map, dataloaders, model, etc.) before any x-precond attempt
    _x_precond_structural_gate = None  # deferred structural-progress check after x-precond is accepted on val_loss
    _x_precond_made_progress = False  # set True when structural change occurs while gate is active
    # (any_sep_progress tracking removed — was assigned but never read)
    any_sep_split = False  # cumulative: True only when an actual add/mult split or leaf prune occurs (not compound-only)

    def _stageA_record_move(
        *,
        move_kind: str,
        parent_ast: Any,
        candidate_ast: Any,
        parent_loss: Any = None,
        candidate_loss: Any = None,
        reason: str = "",
        risk_tags: Optional[Iterable[str]] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record a real Stage-A accepted transaction without changing behavior."""

        nonlocal stageA_move_seq
        stageA_move_seq += 1

        def _safe_float(v):
            try:
                f = float(v)
                return f if math.isfinite(f) else None
            except Exception:
                return None

        def _render(node):
            if node is None:
                return None
            try:
                return _compact_expression_repr(node, max_length=360, y_op_inv=y_op_inv)
            except Exception:
                try:
                    return ast_to_human_readable(node, x_transform_map)
                except Exception:
                    return str(node)

        def _burden(node):
            try:
                atoms = list(collect_nn_atoms(node))
            except Exception:
                atoms = []
            arities: List[int] = []
            for atom_i in atoms:
                try:
                    arities.append(max(0, int(effective_arity(atom_i))))
                except Exception:
                    arities.append(0)
            try:
                support = sorted(int(v) for v in _collect_var_idxs_from_node(node))
            except Exception:
                support = []
            return {
                "nn_total": int(len(atoms)),
                "nn_multivar": int(sum(1 for a in arities if a > 1)),
                "nn_max_arity": int(max(arities) if arities else 0),
                "nn_arities": list(arities),
                "raw_support": list(support),
            }

        tags = {str(t) for t in list(risk_tags or []) if str(t)}
        kind_l = str(move_kind or "").lower()
        if any(tok in kind_l for tok in ("prune", "delete", "projection")):
            tags.add("destructive_prune")
        if "terminal" in kind_l or "closure" in kind_l:
            tags.add("terminal_closure")
        if "compound" in kind_l:
            tags.add("compound_coordinate")
        if "split" in kind_l or "separation" in kind_l:
            tags.add("split_accept")
        if "fit_link" in kind_l or "asinh" in kind_l:
            tags.add("transformed_link")
        if getattr(lm_hp, "fit_y_link", None):
            tags.add("fit_link_active")
        try:
            if bool(x_transform_map):
                tags.add("x_transform_active")
        except Exception:
            pass

        parent_burden = _burden(parent_ast)
        candidate_burden = _burden(candidate_ast)
        rec = {
            "seq": int(stageA_move_seq),
            "move_kind": str(move_kind or "unknown"),
            "outcome": "accepted",
            "reason": str(reason or ""),
            "risk_tags": sorted(tags),
            "y_transform": str(y_transform_name or y_op_str),
            "fit_y_link": getattr(lm_hp, "fit_y_link", None),
            "fit_y_link_scale": _safe_float(getattr(lm_hp, "fit_y_link_scale", 1.0)),
            "parent_loss": _safe_float(parent_loss),
            "candidate_loss": _safe_float(candidate_loss),
            "parent_ast_human": _render(parent_ast),
            "candidate_ast_human": _render(candidate_ast),
            "parent_burden": parent_burden,
            "candidate_burden": candidate_burden,
            "nn_burden_delta": {
                "nn_total": int(candidate_burden["nn_total"] - parent_burden["nn_total"]),
                "nn_multivar": int(candidate_burden["nn_multivar"] - parent_burden["nn_multivar"]),
                "nn_max_arity": int(candidate_burden["nn_max_arity"] - parent_burden["nn_max_arity"]),
            },
            "raw_support_removed": sorted(
                set(parent_burden.get("raw_support", [])) - set(candidate_burden.get("raw_support", []))
            ),
            "raw_support_added": sorted(
                set(candidate_burden.get("raw_support", [])) - set(parent_burden.get("raw_support", []))
            ),
            "details": dict(details or {}),
        }
        provisional_reason = _stageA_provisional_move_reason(
            move_kind=move_kind,
            risk_tags=tags,
            details=rec["details"],
        )
        if provisional_reason:
            rec["provisional"] = True
            rec["provisional_reason"] = str(provisional_reason)
            rec["requires_stageB_confirmation"] = True
            rec["confirmation_status"] = "pending"
            rec["active"] = True
            rec["rollback_available"] = bool(
                rec["details"].get("coe_provisional_budget_admission", False)
            )
            rec["rollback_provenance"] = {
                "kind": (
                    "parent_model_transaction"
                    if rec["rollback_available"]
                    else "parent_ast_snapshot"
                ),
                "parent_ast_human": rec.get("parent_ast_human"),
                "parent_loss": rec.get("parent_loss"),
                "parent_burden": rec.get("parent_burden"),
                "note": (
                    "Parent fitted state is retained through the mandatory next full refit."
                    if rec["rollback_available"]
                    else "AST/loss provenance only; fitted Stage-A model rollback is not cloned."
                ),
            }
            stageA_provisional_commits.append(copy.deepcopy(rec))
        else:
            rec["provisional"] = False
            rec["requires_stageB_confirmation"] = False
        stageA_move_records.append(rec)

    def _stageA_compound_move_details(model_obj: Any, full_compound: bool) -> dict:
        details = {"full_compound": bool(full_compound)}
        for attr, key in (
            ("_stageA_last_compound_old_arity", "old_arity"),
            ("_stageA_last_compound_new_arity", "new_arity"),
            ("_stageA_last_compound_kind", "compound_kind"),
            ("_stageA_last_compound_pattern", "pattern"),
            ("_stageA_last_compound_shadow_requires_payoff", "shadow_requires_payoff"),
            ("_stageA_last_compound_shadow_visible_ast", "shadow_visible_ast"),
            (
                "_stageA_last_compound_coe_provisional_admission",
                "coe_provisional_budget_admission",
            ),
            (
                "_stageA_last_compound_structural_budget_multiplier",
                "structural_budget_multiplier",
            ),
        ):
            try:
                value = getattr(model_obj, attr, None)
                if value is not None:
                    details[key] = copy.deepcopy(value)
            except Exception:
                pass
        try:
            desc = getattr(model_obj, "_stageA_last_compound_replay_descriptor", None)
            if isinstance(desc, dict):
                details["compound_replay_descriptor"] = copy.deepcopy(desc)
        except Exception:
            pass
        try:
            if bool(getattr(model_obj, "_stageA_last_compound_was_scout_replay", False)):
                details["coe_scout_replay"] = True
        except Exception:
            pass
        for attr, key in (
            ("_stageA_last_compound_iso_z_status", "iso_z_status"),
            ("_stageA_last_compound_iso_z_ratio", "iso_z_ratio"),
            ("_stageA_last_compound_iso_z_struct_ratio", "iso_z_struct_ratio"),
            ("_stageA_last_compound_iso_z_noise_ratio", "iso_z_noise_ratio"),
            ("_stageA_last_compound_iso_z_threshold_eff", "iso_z_threshold_eff"),
            ("_stageA_last_compound_iso_z_uncertified", "iso_z_uncertified"),
            ("_stageA_last_compound_proposal_lane_protected", "proposal_lane_protected"),
        ):
            try:
                value = getattr(model_obj, attr, None)
                if value is not None:
                    details[key] = copy.deepcopy(value)
            except Exception:
                pass
        if bool(details.get("iso_z_uncertified", False)):
            details["null_verified"] = False
        return details

    def _stageA_parent_context_key(node: Any) -> str:
        try:
            return ast_to_human_readable(node, x_transform_map)
        except Exception:
            try:
                return str(node)
            except Exception:
                return "<unknown_parent>"

    def _stageA_early_compound_transaction_key(
        *,
        parent_ast: Any,
        z_var_idxs: Iterable[int],
        z_exponents: Iterable[int],
        remaining_var_idxs: Iterable[int],
        kind: str = "early_compound",
    ) -> tuple:
        return (
            str(kind),
            str(y_transform_name or y_op_str),
            _stageA_parent_context_key(parent_ast),
            tuple(int(v) for v in z_var_idxs),
            tuple(int(v) for v in z_exponents),
            tuple(int(v) for v in remaining_var_idxs),
        )

    def _stageA_record_rejected_transaction(
        *,
        key: tuple,
        move_kind: str,
        parent_ast: Any,
        candidate_ast: Any,
        failure_summary: Dict[str, Any],
        details: Optional[Dict[str, Any]] = None,
        move_seq: Optional[int] = None,
    ) -> None:
        stageA_rejected_transactions.add(tuple(key))
        record = {
            "move_kind": str(move_kind),
            "reason": "full_refit_failed_after_provisional_commit",
            "failure": dict(failure_summary or {}),
            "parent_ast_human": _stageA_parent_context_key(parent_ast),
            "candidate_ast_human": _stageA_parent_context_key(candidate_ast),
            "transaction_key": [str(x) for x in tuple(key)],
            "details": dict(details or {}),
        }
        stageA_rejection_records.append(record)
        if move_seq is not None:
            for move in stageA_move_records:
                if int(move.get("seq", -1)) == int(move_seq):
                    move["outcome"] = "rolled_back"
                    move["active"] = False
                    move["rollback"] = dict(failure_summary or {})
            for commit in stageA_provisional_commits:
                if int(commit.get("seq", -1)) == int(move_seq):
                    commit["active"] = False
                    commit["confirmation_status"] = "rolled_back_after_failed_refit"
                    commit["rollback"] = dict(failure_summary or {})

    def _stageA_compound_transaction_key(
        *,
        move_kind: str,
        parent_ast: Any,
        candidate_ast: Any,
        details: Optional[Dict[str, Any]] = None,
    ) -> tuple:
        details = dict(details or {})
        try:
            cand_key = str(_stageA_ast_fingerprint(candidate_ast))
        except Exception:
            cand_key = _stageA_parent_context_key(candidate_ast)
        return (
            "stageA_compound",
            str(move_kind),
            str(y_transform_name or y_op_str),
            _stageA_parent_context_key(parent_ast),
            cand_key,
            str(details.get("compound_kind", "")),
            str(details.get("old_arity", "")),
            str(details.get("new_arity", "")),
            repr(tuple(details.get("pattern", ()) or ())),
            str(bool(details.get("coe_scout_replay", False))),
            str(bool(details.get("visible_buckingham_1d_prefactor", False))),
            str(bool(details.get("soft_monomial_compound", False))),
        )

    def _stageA_begin_pending_full_refit_transaction(
        *,
        move_kind: str,
        parent_ast: Any,
        parent_model: Any,
        parent_models_multi: Any,
        parent_val_loss: Any,
        parent_val_losses: Any,
        candidate_ast: Any,
        candidate_loss: Any,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        nonlocal pending_stageA_full_refit_transaction

        try:
            if not collect_nn_atoms(candidate_ast):
                return
        except Exception:
            return
        details = dict(details or {})
        key = _stageA_compound_transaction_key(
            move_kind=move_kind,
            parent_ast=parent_ast,
            candidate_ast=candidate_ast,
            details=details,
        )
        if key in stageA_rejected_transactions:
            return

        try:
            cand_loss_value = float(candidate_loss)
            if not math.isfinite(cand_loss_value):
                cand_loss_value = None
        except Exception:
            cand_loss_value = None
        key_record = {
            "key": key,
            "move_kind": str(move_kind),
            "parent_ast": parent_ast,
            "candidate_ast": candidate_ast,
            "candidate_loss": cand_loss_value,
            "details": copy.deepcopy(details),
            "move_seq": int(stageA_move_seq),
        }
        if pending_stageA_full_refit_transaction is None:
            pending_stageA_full_refit_transaction = {
                "parent_ast": parent_ast,
                "parent_model": parent_model,
                "parent_models_multi": (
                    list(parent_models_multi)
                    if parent_models_multi is not None
                    else None
                ),
                "parent_val_loss": parent_val_loss,
                "parent_val_losses": (
                    list(parent_val_losses)
                    if parent_val_losses is not None
                    else None
                ),
                "parent_dual_layer_used": dual_layer_used,
                "parent_i": int(i),
                "parent_x_transform_map": dict(x_transform_map),
                "parent_train_dl": datagen_train_noshuffle,
                "parent_val_dl": datagen_val_noshuffle,
                "parent_feats": feats,
                "parent_scale_specs": scale_specs,
                "parent_parity_specs": parity_specs,
                "parent_trig_spec": trig_spec,
                "parent_invariance_feats": invariance_feats,
                "candidate_ast": candidate_ast,
                "candidate_loss": candidate_loss,
                "moves": [key_record],
            }
        else:
            pending_stageA_full_refit_transaction["candidate_ast"] = candidate_ast
            pending_stageA_full_refit_transaction["candidate_loss"] = candidate_loss
            pending_stageA_full_refit_transaction.setdefault("moves", []).append(key_record)

    # Optional dimensional-analysis support inside Stage A.
    units_spec = None
    units_raw_x_dims = None
    if bool(enforce_units) and (units_payload is not None):
        try:
            from nestynet_sr.sr_core.units import UnitsSpec

            us = units_payload.get("unit_system")
            units_raw_x_dims = units_payload.get("x_dims")
            y_dim_units = units_payload.get("y_dim")
            free_const_dims = units_payload.get("free_const_dims", {})
            fixed_const_dims = units_payload.get("fixed_const_dims", {})
            fixed_const_values = units_payload.get("fixed_const_values", {})
            fixed_const_mode = units_payload.get("fixed_const_mode", "strict")

            if us is not None and units_raw_x_dims is not None and y_dim_units is not None:
                free_const_scope = units_payload.get("free_const_scope", {})
                units_spec = UnitsSpec(
                    unit_system=us,
                    x_dims=tuple(units_raw_x_dims),
                    y_dim=tuple(y_dim_units),
                    y_transform_name=str(y_transform_name),
                    free_const_dims=dict(free_const_dims),
                    free_const_scope=dict(free_const_scope),
                    fixed_const_dims=dict(fixed_const_dims),
                    fixed_const_values=dict(fixed_const_values),
                    fixed_const_mode=str(fixed_const_mode),
                    policy=str(units_policy),
                    nn_semantics=str(nn_units_semantics),
                )
        except Exception as e:
            print(f"[Units] Warning: Stage A could not initialise units spec: {e}")
            units_spec = None
            units_raw_x_dims = None

    def _internal_x_dims_for_map(xmap):
        if units_spec is None or units_raw_x_dims is None:
            return None
        try:
            from .xcoord import XCoordSystem

            xcoords = XCoordSystem.from_map(xmap or {}, Nx_raw=Nxvars)
            return xcoords.internal_x_dims(units_spec.unit_system, units_raw_x_dims)
        except Exception:
            raise

    def _refresh_units_spec_for_xmap(xmap):
        nonlocal units_spec
        if units_spec is None:
            return
        try:
            internal = _internal_x_dims_for_map(xmap)
            if internal is None:
                return
            from dataclasses import replace

            units_spec = replace(units_spec, x_dims=tuple(internal))
        except Exception as e:
            print(f"[Units] Warning: could not update internal x units for x-transform map: {e}")

    def _restore_x_precond_state(saved_state):
        """Restore all state from a saved x-preconditioning snapshot."""
        return (
            saved_state["x_transform_map"],
            saved_state["train_dl"],
            saved_state["val_dl"],
            saved_state["current_ast"],
            saved_state["model"],
            saved_state["i"],
            saved_state["current_val_loss"],
            saved_state.get("dual_layer_used", False),
            saved_state.get("feats", []),
            saved_state.get("scale_specs", []),
            saved_state.get("parity_specs", []),
            saved_state.get("trig_spec", None),
            saved_state.get("invariance_feats", None),
        )

    # Optional x-preconditioning loop: if Stage A stalls before fully separating,
    # we can retry in transformed x-coordinates (e.g., reciprocal / trig substitutions).
    x_precond_enable = bool(getattr(search_hp, 'x_precondition_enable', True))
    x_precond_max_extra_passes = int(getattr(search_hp, 'x_precondition_max_extra_passes', 2))
    x_precond_scaling_tol = float(getattr(search_hp, 'x_precondition_scaling_tol', 0.35))
    x_precond_applied = set()  # e.g., {'recip', 'trig'}
    x_precond_attempts = 0
    if mode != "full":
        x_precond_enable = False
    # Keep Stage-A multi path conservative for now: avoid x-preconditioning retries.
    if is_multi:
        x_precond_enable = False

    # --- Derive MAD-based loss scaling for this y_op ---
    use_mad_units = getattr(lm_hp, "loss_in_MAD_units", False)
    if use_mad_units:
        fit_link_name = getattr(lm_hp, "fit_y_link", None)
        fit_link_scale = getattr(lm_hp, "fit_y_link_scale", 1.0)
        if is_multi:
            per_scales = []
            for dl_i in train_loaders_all:
                _, y_mad_tf_i = _compute_y_med_mad_from_loader(
                    dl_i,
                    device,
                    fit_y_link=fit_link_name,
                    fit_y_link_scale=fit_link_scale,
                )
                if (y_mad_tf_i is not None) and (y_mad_tf_i > 0.0):
                    per_scales.append(float(y_mad_tf_i) ** 2)
                else:
                    per_scales.append(1.0)
            loss_scale = _aggregate_losses(per_scales, mode=agg_mode, weights=agg_weights)
            if (not math.isfinite(loss_scale)) or loss_scale <= 0.0:
                loss_scale = 1.0
                print("Warning: aggregated MAD(φ(y)) scale invalid; using unscaled LM thresholds.")
            else:
                print(
                    f"[Stage A multi] Using aggregated MAD-normalised loss scale={loss_scale:.3g} "
                    f"({len(per_scales)} datasets, mode={agg_mode})"
                )
        else:
            _, y_mad_tf = _compute_y_med_mad_from_loader(
                datagen_train_noshuffle,
                device,
                fit_y_link=fit_link_name,
                fit_y_link_scale=fit_link_scale,
            )
            if (y_mad_tf is None) or (y_mad_tf <= 0.0):
                loss_scale = 1.0
                print("Warning: MAD(φ(y)) non-positive or undefined; using unscaled LM thresholds.")
            else:
                loss_scale = y_mad_tf**2
                link = canonical_fit_link_name(fit_link_name)
                mad_label = "MAD(φ(y))" if link is None else f"MAD({describe_fit_link(link, fit_link_scale)})"
                print(
                    f"Using MAD-normalised LM thresholds for y_op {y_op_str}: "
                    f"{mad_label}≈{y_mad_tf:.3g}, scale={loss_scale:.3g}"
                )
    else:
        loss_scale = 1.0

    loss_target_eff = 0.0
    loss_acceptable_eff_init = 0.0
    accept_threshold_eff_cand = 0.0
    global_ceil = None

    def _has_explicit_acceptance_noise_floor() -> bool:
        try:
            raw = float(getattr(lm_hp, "acceptance_noise_floor_raw", None))
            if math.isfinite(raw) and raw >= 0.0:
                return True
        except Exception:
            pass
        try:
            base = float(getattr(lm_hp, "acceptance_noise_floor", None))
            if math.isfinite(base) and base >= 0.0:
                return True
        except Exception:
            pass
        return False

    _noise_floor_dynamic_enabled = (
        (lm_hp is not None)
        and (y_raw_full is not None)
        and (noise_sigma_y is not None)
        and (float(noise_sigma_y) > 0.0)
        and (not _has_explicit_acceptance_noise_floor())
    )
    _noise_floor_cache: Dict[Tuple[Optional[str], float], Optional[float]] = {}
    _noise_floor_logged: set = set()

    def _current_acceptance_noise_floor_raw() -> float:
        base_floor = (
            0.0
            if _noise_floor_dynamic_enabled
            else _resolve_acceptance_noise_floor_raw(lm_hp, loss_scale)
        )
        if not _noise_floor_dynamic_enabled:
            return float(base_floor)

        link_name = canonical_fit_link_name(getattr(lm_hp, "fit_y_link", None))
        link_scale = float(getattr(lm_hp, "fit_y_link_scale", 1.0))
        cache_key = (link_name, link_scale)
        est_floor = _noise_floor_cache.get(cache_key, None)
        if cache_key not in _noise_floor_cache:
            est_floor = _estimate_transform_noise_floor_raw(
                y_raw_full,
                y_op,
                noise_sigma_y,
                fit_link=link_name,
                fit_link_scale=link_scale,
                n_mc=noise_floor_mc_samples,
            )
            _noise_floor_cache[cache_key] = est_floor
        if cache_key not in _noise_floor_logged:
            label = y_transform_name or y_op_str
            if est_floor is None:
                print(
                    f"[Noise] y-space={label}, fit-link={describe_fit_link(link_name, link_scale)}: "
                    f"could not estimate a stable loss floor; leaving thresholds unchanged."
                )
            else:
                print(
                    f"[Noise] y-space={label}, fit-link={describe_fit_link(link_name, link_scale)}: "
                    f"sigma_y={float(noise_sigma_y):.3g}, floor_raw={float(est_floor):.3g}"
                )
            _noise_floor_logged.add(cache_key)

        if est_floor is None:
            try:
                lm_hp.acceptance_noise_floor_raw = None
            except Exception:
                pass
            return float(base_floor)

        try:
            lm_hp.acceptance_noise_floor_raw = float(est_floor)
        except Exception:
            pass
        return float(est_floor)

    def _current_stageA_yspace_noise_floor_raw() -> float:
        """Noise floor in model-output space before any LM-only fit-link."""
        if not _noise_floor_dynamic_enabled:
            if canonical_fit_link_name(getattr(lm_hp, "fit_y_link", None)) is not None:
                return 0.0
            return float(_resolve_acceptance_noise_floor_raw(lm_hp, loss_scale))

        cache_key = ("__stageA_yspace__", 1.0)
        est_floor = _noise_floor_cache.get(cache_key, None)
        if cache_key not in _noise_floor_cache:
            est_floor = _estimate_transform_noise_floor_raw(
                y_raw_full,
                y_op,
                noise_sigma_y,
                fit_link=None,
                fit_link_scale=1.0,
                n_mc=noise_floor_mc_samples,
            )
            _noise_floor_cache[cache_key] = est_floor
        if est_floor is None:
            return 0.0
        return float(est_floor)

    def _refresh_effective_loss_thresholds():
        nonlocal loss_target_eff
        nonlocal loss_acceptable_eff_init
        nonlocal accept_threshold_eff_cand
        nonlocal global_ceil
        nonlocal acceptance_noise_floor_raw

        loss_target_eff = lm_hp.loss_target * loss_scale
        loss_acceptable_eff_init = lm_hp.loss_acceptable * loss_scale
        accept_threshold_eff_cand = search_hp.loss_acceptable * loss_scale
        acceptance_noise_floor_raw = _current_acceptance_noise_floor_raw()
        (
            loss_target_eff,
            loss_acceptable_eff_init,
            accept_threshold_eff_cand,
        ) = _apply_noise_floor_to_acceptance_thresholds(
            loss_target_raw=loss_target_eff,
            loss_acceptable_raw=loss_acceptable_eff_init,
            accept_threshold_raw=accept_threshold_eff_cand,
            noise_floor_raw=acceptance_noise_floor_raw,
        )
        try:
            lm_hp.stageA_yspace_noise_floor_raw = float(_current_stageA_yspace_noise_floor_raw())
        except Exception:
            lm_hp.stageA_yspace_noise_floor_raw = 0.0
        try:
            n_eff = float(sum(_loader_size(dl) for dl in val_loaders_all))
            lm_hp.acceptance_noise_n_eff = n_eff if n_eff > 0.0 else None
        except Exception:
            lm_hp.acceptance_noise_n_eff = None

        # Global best-loss ceiling: prevent accepting candidates much worse than
        # the best fit across all y-transforms. Converts global_best from base
        # units to this transform's native loss space, then applies the same
        # max_worsening_factor.
        if global_best_val_loss_base is not None:
            _mwf = float(getattr(search_hp, "max_worsening_factor", 100.0))
            global_ceil = global_best_val_loss_base * loss_scale * _mwf
        else:
            global_ceil = None

    acceptance_noise_floor_raw = 0.0
    _refresh_effective_loss_thresholds()

    i = 0
    # For overlapping (partial) separations, require them to reach a much tighter
    # LM accept_threshold, relative to the baseline current model.
    partial_max_worsening_factor = getattr(search_hp, "partial_max_worsening_factor", 3.0)
    separable = True
    separability_success = False
    rest_add = None
    rest_mult = None
    model = None
    current_ast = initial_ast
    best_initial_loss = None
    stageA_initial_report_val_loss = None
    stageA_initial_report_val_losses = None
    stageA_initial_report_n_params = None
    current_val_loss = None  # persistent val-loss for the currently accepted model
    dual_layer_used = False  # persistent architecture flag for the currently accepted model
    last_resort_suggested = False
    full_compound_solved = False  # True if compound covering ALL vars achieved target loss
    early_compound_checkpoint = None  # Validated checkpoint from Early Compound detection
    last_good_stageA_checkpoint = None  # Latest Stage-A state meeting the normal validation threshold
    last_good_stageA_checkpoint_seq = 0

    # These hints are discovered opportunistically (only for y_op None).
    # Initialize here so later logic can safely reference them.
    feats = []
    scale_specs = []
    parity_specs = []
    trig_spec = None
    invariance_feats = None
    trig_scale_specs = []
    trig_axis_specs_all = []
    stagea_best_sep_metric = None
    stagea_best_sep_score = 0.0
    stagea_best_split_score = 0.0
    stagea_sep_candidates_seen = 0
    stagea_split_accept_count = 0
    fit_y_link_tested = False
    _initial_fit_attempt = 0   # retry counter for unlucky initializations (i==0 only)
    _max_initial_fit_attempts = 2  # one normal attempt plus one random-initialization restart
    _stageA_initial_restart_used = False
    _stageA_initial_restart_active = False
    _stageA_plain_random_branch_active = False
    _stageA_plain_random_branch_prev_canonical = None
    _stageA_plain_random_branch_prev_evidence = None
    stageA_fit_link_certificate: Dict[str, Any] = {
        "fit_y_link": canonical_fit_link_name(getattr(lm_hp, "fit_y_link", None)),
        "fit_y_link_scale": float(getattr(lm_hp, "fit_y_link_scale", 1.0)),
        "status": (
            "identity"
            if canonical_fit_link_name(getattr(lm_hp, "fit_y_link", None)) is None
            else "unchecked"
        ),
        "transformed_loss_ok": False,
        "original_y_certified": canonical_fit_link_name(getattr(lm_hp, "fit_y_link", None)) is None,
        "original_y_val_loss": None,
        "original_y_allowed_loss": None,
        "reason": "initial",
    }

    def _initial_fit_call_with_restart_policy(func, *args, **kwargs):
        """Disable canonical init and evidence for the one random-restart initial fit."""
        if not _stageA_initial_restart_active:
            return func(*args, **kwargs)
        prev_canonical = bool(getattr(lm_hp, "canonical_init", False))
        prev_evidence = bool(getattr(lm_hp, "evidence_enable", False))
        try:
            lm_hp.canonical_init = False
            lm_hp.evidence_enable = False
            return func(*args, **kwargs)
        finally:
            lm_hp.canonical_init = prev_canonical
            lm_hp.evidence_enable = prev_evidence

    def _activate_plain_random_branch_after_initial_restart(reason: str) -> None:
        """Keep the successful random/no-evidence branch policy until structure changes."""
        nonlocal _stageA_plain_random_branch_active
        nonlocal _stageA_plain_random_branch_prev_canonical
        nonlocal _stageA_plain_random_branch_prev_evidence

        if _stageA_plain_random_branch_active:
            return
        _stageA_plain_random_branch_prev_canonical = bool(getattr(lm_hp, "canonical_init", False))
        _stageA_plain_random_branch_prev_evidence = bool(getattr(lm_hp, "evidence_enable", False))
        lm_hp.canonical_init = False
        lm_hp.evidence_enable = False
        _stageA_plain_random_branch_active = True
        print(
            f"{YELLOW}[Stage A] {reason}; keeping canonical initialization and evidence disabled "
            f"until the first confirmed structural Stage-A move.{RESET}"
        )

    def _restore_plain_random_branch_policy(reason: str) -> None:
        nonlocal _stageA_plain_random_branch_active
        nonlocal _stageA_plain_random_branch_prev_canonical
        nonlocal _stageA_plain_random_branch_prev_evidence

        if not _stageA_plain_random_branch_active:
            return
        if _stageA_plain_random_branch_prev_canonical is not None:
            lm_hp.canonical_init = bool(_stageA_plain_random_branch_prev_canonical)
        if _stageA_plain_random_branch_prev_evidence is not None:
            lm_hp.evidence_enable = bool(_stageA_plain_random_branch_prev_evidence)
        _stageA_plain_random_branch_active = False
        _stageA_plain_random_branch_prev_canonical = None
        _stageA_plain_random_branch_prev_evidence = None
        print(
            f"{YELLOW}[Stage A] {reason}; restoring canonical initialization/evidence policy "
            f"for subsequent fits.{RESET}"
        )

    def _clear_plain_random_branch_after_structural_accept(reason: str) -> None:
        _restore_plain_random_branch_policy(f"Confirmed structural move ({reason})")

    def _try_asinh_fit_with_restart_policy(**kwargs):
        return _initial_fit_call_with_restart_policy(_try_asinh_fit, **kwargs)

    def _attach_fit_link_certificate(target_model=None) -> None:
        """Attach current fit-link branch certification metadata to model(s)."""
        payload = dict(stageA_fit_link_certificate)
        targets = []
        if target_model is not None:
            targets.append(target_model)
        if model is not None and all(id(model) != id(t) for t in targets):
            targets.append(model)
        if models_multi is not None:
            for m_i in models_multi:
                if m_i is not None and all(id(m_i) != id(t) for t in targets):
                    targets.append(m_i)
        for m_i in targets:
            try:
                setattr(m_i, "_stageA_fit_link_certificate", dict(payload))
                setattr(m_i, "_stageA_original_y_certified", bool(payload.get("original_y_certified", False)))
                setattr(m_i, "_stageA_fit_link_branch_status", str(payload.get("status", "unknown")))
                setattr(m_i, "_stageA_original_y_val_loss", payload.get("original_y_val_loss", None))
                setattr(m_i, "_stageA_original_y_allowed_loss", payload.get("original_y_allowed_loss", None))
            except Exception:
                pass

    def _set_fit_link_certificate(
        *,
        status: str,
        transformed_loss_ok: bool,
        original_y_certified: bool,
        reason: str,
        transformed_val_loss=None,
        original_y_val_loss=None,
        original_y_allowed_loss=None,
        original_y_D_ref=None,
    ) -> None:
        nonlocal stageA_fit_link_certificate
        link = canonical_fit_link_name(getattr(lm_hp, "fit_y_link", None))
        stageA_fit_link_certificate = {
            "fit_y_link": link,
            "fit_y_link_scale": float(getattr(lm_hp, "fit_y_link_scale", 1.0)),
            "status": str(status),
            "transformed_loss_ok": bool(transformed_loss_ok),
            "transformed_val_loss": (
                None if transformed_val_loss is None else float(transformed_val_loss)
            ),
            "original_y_certified": bool(original_y_certified),
            "original_y_val_loss": (
                None if original_y_val_loss is None else float(original_y_val_loss)
            ),
            "original_y_allowed_loss": (
                None if original_y_allowed_loss is None else float(original_y_allowed_loss)
            ),
            "original_y_D_ref": (
                None if original_y_D_ref is None else float(original_y_D_ref)
            ),
            "reason": str(reason),
        }
        _attach_fit_link_certificate()

    def _refresh_fit_link_original_y_certificate(reason: str, *, quiet: bool = False) -> bool:
        """Recheck whether the active fit-link branch is certified in original-y space."""
        link = canonical_fit_link_name(getattr(lm_hp, "fit_y_link", None))
        transformed_loss = current_val_loss
        try:
            transformed_ok = (
                transformed_loss is not None
                and math.isfinite(float(transformed_loss))
                and float(transformed_loss) < float(loss_acceptable_eff_init)
            )
        except Exception:
            transformed_ok = False

        if link is None:
            _set_fit_link_certificate(
                status="identity",
                transformed_loss_ok=bool(transformed_ok),
                original_y_certified=True,
                reason=reason,
                transformed_val_loss=transformed_loss,
            )
            return True

        if link != "asinh":
            _set_fit_link_certificate(
                status="unchecked",
                transformed_loss_ok=bool(transformed_ok),
                original_y_certified=False,
                reason=f"{reason}: unsupported fit-link certificate",
                transformed_val_loss=transformed_loss,
            )
            if not quiet:
                print(
                    f"{YELLOW}[Stage A] fit-link branch '{link}' is transformed-space only; "
                    "original-y certification is unchecked."
                    f"{RESET}"
                )
            return False

        if model is None:
            _set_fit_link_certificate(
                status="search_scaffold",
                transformed_loss_ok=bool(transformed_ok),
                original_y_certified=False,
                reason=f"{reason}: no model",
                transformed_val_loss=transformed_loss,
            )
            return False

        was_certified = bool(stageA_fit_link_certificate.get("original_y_certified", False))
        try:
            if is_multi:
                y_mses = []
                y_allows = []
                d_refs = []
                ok_all = True
                per_models = list(models_multi) if models_multi is not None else [model]
                per_losses = (
                    list(current_val_losses)
                    if current_val_losses is not None and len(current_val_losses) == len(per_models)
                    else [transformed_loss for _ in per_models]
                )
                for di, tm in enumerate(per_models):
                    dl_i = val_loaders_all[di] if di < len(val_loaders_all) else datagen_val_noshuffle
                    loss_i = per_losses[di] if di < len(per_losses) else transformed_loss
                    ok_i, y_mse_i, y_allow_i, d_ref_i, _ = _check_asinh_yspace_sanity(
                        model=tm,
                        dl_val=dl_i,
                        device=device,
                        asinh_loss=float(loss_i),
                        lm_hp=lm_hp,
                        base_model=None,
                    )
                    ok_all = bool(ok_all and ok_i)
                    y_mses.append(float(y_mse_i))
                    y_allows.append(float(y_allow_i))
                    d_refs.append(float(d_ref_i))
                y_mse = _aggregate_losses(y_mses, mode=agg_mode, weights=agg_weights)
                y_allow = _aggregate_losses(y_allows, mode=agg_mode, weights=agg_weights)
                d_ref = _aggregate_losses(d_refs, mode=agg_mode, weights=agg_weights)
                ok = bool(ok_all)
            else:
                ok, y_mse, y_allow, d_ref, _ = _check_asinh_yspace_sanity(
                    model=model,
                    dl_val=datagen_val_noshuffle,
                    device=device,
                    asinh_loss=float(transformed_loss),
                    lm_hp=lm_hp,
                    base_model=None,
                )
        except Exception as exc:
            _set_fit_link_certificate(
                status="search_scaffold",
                transformed_loss_ok=bool(transformed_ok),
                original_y_certified=False,
                reason=f"{reason}: certification error {type(exc).__name__}",
                transformed_val_loss=transformed_loss,
            )
            if not quiet:
                print(f"{YELLOW}[Stage A] fit-link original-y certification check failed: {exc}{RESET}")
            return False

        status = "original_y_certified" if bool(ok) else "search_scaffold"
        _set_fit_link_certificate(
            status=status,
            transformed_loss_ok=bool(transformed_ok),
            original_y_certified=bool(ok),
            reason=reason,
            transformed_val_loss=transformed_loss,
            original_y_val_loss=float(y_mse),
            original_y_allowed_loss=float(y_allow),
            original_y_D_ref=float(d_ref),
        )
        if bool(ok):
            if (not was_certified) or not quiet:
                print(
                    f"{GREEN}[Stage A] fit-link original-y certified after {reason}: "
                    f"y-MSE={float(y_mse):.3e} <= allowed={float(y_allow):.3e}.{RESET}"
                )
        elif not quiet:
            print(
                f"{YELLOW}[Stage A] fit-link branch remains a search scaffold after {reason}: "
                f"original-y y-MSE={float(y_mse):.3e} > allowed={float(y_allow):.3e}.{RESET}"
            )
        return bool(ok)

    def _reset_loss_scale_for_current_fit_link():
        nonlocal loss_scale

        if not use_mad_units:
            loss_scale = 1.0
            _refresh_effective_loss_thresholds()
            return

        fit_link_name = getattr(lm_hp, "fit_y_link", None)
        fit_link_scale = getattr(lm_hp, "fit_y_link_scale", 1.0)
        if is_multi:
            per_scales = []
            for dl_i in train_loaders_all:
                _, y_mad_tf_i = _compute_y_med_mad_from_loader(
                    dl_i,
                    device,
                    fit_y_link=fit_link_name,
                    fit_y_link_scale=fit_link_scale,
                )
                if y_mad_tf_i is not None and y_mad_tf_i > 0.0:
                    per_scales.append(float(y_mad_tf_i) ** 2)
                else:
                    per_scales.append(1.0)
            loss_scale = _aggregate_losses(per_scales, mode=agg_mode, weights=agg_weights)
        else:
            _, y_mad_tf = _compute_y_med_mad_from_loader(
                datagen_train_noshuffle,
                device,
                fit_y_link=fit_link_name,
                fit_y_link_scale=fit_link_scale,
            )
            loss_scale = float(y_mad_tf) ** 2 if (y_mad_tf is not None and y_mad_tf > 0.0) else 1.0
        if (not math.isfinite(loss_scale)) or loss_scale <= 0.0:
            loss_scale = 1.0
        _refresh_effective_loss_thresholds()

    def _start_initial_random_restart(reason: str, val_loss: Optional[float] = None) -> bool:
        nonlocal _stageA_initial_restart_used
        nonlocal _stageA_initial_restart_active
        nonlocal model
        nonlocal models_multi
        nonlocal current_ast
        nonlocal current_val_loss
        nonlocal current_val_losses
        nonlocal best_initial_loss
        nonlocal dual_layer_used
        nonlocal fit_y_link_tested

        if not _stageA_initial_fit_restart_allowed(
            y_op_is_identity=(y_op is None),
            is_multi=bool(is_multi),
            skip_initial_fit=bool(skip_initial_fit),
            restart_used=bool(_stageA_initial_restart_used),
            has_previous_model=(model_prev is not None),
            fit_y_link_active=(getattr(lm_hp, "fit_y_link", None) is not None),
        ):
            return False
        if x_precond_saved_state is not None:
            return False

        _stageA_initial_restart_used = True
        _stageA_initial_restart_active = True
        val_note = ""
        try:
            if val_loss is not None and math.isfinite(float(val_loss)):
                val_note = f" (val_loss={float(val_loss):.4e})"
        except Exception:
            val_note = ""
        restart_notes = []
        if bool(getattr(lm_hp, "canonical_init", False)):
            restart_notes.append("canonical initialization disabled")
        else:
            restart_notes.append("a fresh random initialization")
        if bool(getattr(lm_hp, "evidence_enable", False)):
            restart_notes.append("evidence disabled")
        restart_note = "with " + " and ".join(restart_notes)
        print(
            f"{YELLOW}[Stage A] {reason}{val_note}; retrying the initial identity fit once "
            f"{restart_note} before saving the reference model "
            f"(attempt {_max_initial_fit_attempts}/{_max_initial_fit_attempts}).{RESET}"
        )

        model = None
        models_multi = None
        current_ast = initial_ast
        current_val_loss = None
        current_val_losses = None
        best_initial_loss = None
        dual_layer_used = False
        lm_hp.fit_y_link = None
        lm_hp.fit_y_link_scale = 1.0
        _reset_loss_scale_for_current_fit_link()
        fit_y_link_tested = False
        return True

    def _train_initial_models(models_in: List[torch.nn.Module]):
        """Train one model per dataset; return aggregate + per-dataset losses."""
        if len(models_in) != len(train_loaders_all):
            raise ValueError(
                f"models/train_loader length mismatch: {len(models_in)} vs {len(train_loaders_all)}"
            )
        if len(models_in) == 1:
            m0 = models_in[0]
            best_val_loss_0, best_train_loss_0, best_val_p_0, lm_opt_0 = fit_initial_model_with_tournament(
                m0,
                train_loaders_all[0],
                val_loaders_all[0],
                epochs=lm_hp.epochs,
                LM_strategy=lm_hp.strategy,
                nval_patience=lm_hp.nval_patience,
                loss_target=loss_target_eff,
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
                lm_hp=lm_hp,
            )
            lm_opt_0._update_param_groups(best_val_p_0)
            return (
                float(best_val_loss_0),
                float(best_train_loss_0),
                [float(best_val_loss_0)],
                [float(best_train_loss_0)],
                models_in,
            )

        per_val_losses: List[float] = []
        per_train_losses: List[float] = []
        for di, m_i in enumerate(models_in):
            best_val_loss_i, best_train_loss_i, best_val_p_i, lm_opt_i = train_initial_model(
                m_i,
                train_loaders_all[di],
                val_loaders_all[di],
                epochs=lm_hp.epochs,
                LM_strategy=lm_hp.strategy,
                nval_patience=lm_hp.nval_patience,
                loss_target=loss_target_eff,
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
            if best_val_p_i is not None:
                lm_opt_i._update_param_groups(best_val_p_i)
            per_val_losses.append(float(best_val_loss_i))
            per_train_losses.append(float(best_train_loss_i))
        val_agg = _aggregate_losses(per_val_losses, mode=agg_mode, weights=agg_weights)
        train_agg = _aggregate_losses(per_train_losses, mode=agg_mode, weights=agg_weights)
        return float(val_agg), float(train_agg), per_val_losses, per_train_losses, models_in

    def _train_candidate_models(models_in: List[torch.nn.Module], accept_threshold: float,
                                extra_train_factories=None):
        """Train candidate models and decide acceptance from aggregated validation loss."""
        if len(models_in) != len(train_loaders_all):
            raise ValueError(
                f"models/train_loader length mismatch: {len(models_in)} vs {len(train_loaders_all)}"
            )
        if len(models_in) == 1:
            max_train_degradation = float(
                getattr(search_hp, "max_train_degradation", 100.0)
            )
            lane_train_loss_cap = (
                float("inf")
                if best_train_loss_initial is None or best_train_loss_initial <= 0
                else max(
                    max_train_degradation * best_train_loss_initial,
                    loss_target_eff,
                )
            )
            accepted, best_val_loss_0, best_train_loss_0, best_val_p_0, lm_opt_0 = fit_stageA_candidate_with_tournament(
                models_in[0],
                train_loaders_all[0],
                val_loaders_all[0],
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
                extra_train_factories=extra_train_factories,
                y_op=y_op,
                y_op_inv=y_op_inv,
                max_lane_train_loss=lane_train_loss_cap,
                lm_hp=lm_hp,
            )
            if best_val_p_0 is not None:
                lm_opt_0._update_param_groups(best_val_p_0)
            return (
                bool(accepted),
                float(best_val_loss_0),
                float(best_train_loss_0),
                [float(best_val_loss_0)],
                [float(best_train_loss_0)],
                models_in,
            )

        per_val_losses: List[float] = []
        per_train_losses: List[float] = []
        for di, m_i in enumerate(models_in):
            _, best_val_loss_i, best_train_loss_i, best_val_p_i, lm_opt_i = train_candidate_model(
                m_i,
                train_loaders_all[di],
                val_loaders_all[di],
                epochs=lm_hp.epochs,
                LM_strategy=lm_hp.strategy,
                nval_patience=lm_hp.nval_patience,
                loss_target=loss_target_eff,
                accept_threshold=float("inf"),
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
            if best_val_p_i is not None:
                lm_opt_i._update_param_groups(best_val_p_i)
            per_val_losses.append(float(best_val_loss_i))
            per_train_losses.append(float(best_train_loss_i))
        val_agg = _aggregate_losses(per_val_losses, mode=agg_mode, weights=agg_weights)
        train_agg = _aggregate_losses(per_train_losses, mode=agg_mode, weights=agg_weights)
        accepted_agg = bool(float(val_agg) < float(accept_threshold))
        return accepted_agg, float(val_agg), float(train_agg), per_val_losses, per_train_losses, models_in

    def _train_overlap_candidate_with_gauge_continuation(
        models_in: List[torch.nn.Module],
        accept_threshold: float,
        candidate_ast_local,
        g1_local,
        g2_local,
        parent_tag_local,
        op_local,
    ):
        """Warm-start overlap candidates without gauge, then require a non-zero gauge stage."""
        from nestynet_sr.adaptors.gauge_fix_adaptor import (
            gauge_fix_diagnostic,
            gauge_fix_metrics,
        )

        def _build_overlap_gauge_factories(weight: float):
            if len(models_in) != 1:
                return None
            if op_local is torch.add:
                return _build_additive_gauge_fix_factories(
                    models_in[0],
                    candidate_ast_local,
                    g1_local,
                    g2_local,
                    parent_tag_local,
                    datagen_train_noshuffle,
                    device,
                    dtype,
                    weight=weight,
                ) or None
            if op_local in (torch.mul, torch.multiply):
                return _build_multiplicative_gauge_fix_factories(
                    models_in[0],
                    candidate_ast_local,
                    g1_local,
                    g2_local,
                    parent_tag_local,
                    datagen_train_noshuffle,
                    device,
                    dtype,
                    weight=weight,
                ) or None
            return None

        plain_result = _train_candidate_models(
            models_in,
            accept_threshold,
            extra_train_factories=None,
        )
        (
            accepted_plain,
            best_val_loss_plain,
            best_train_loss_plain,
            per_val_losses_plain,
            per_train_losses_plain,
            models_in,
        ) = plain_result

        if (
            len(models_in) != 1
            or not bool(getattr(lm_hp, "overlap_gauge_continuation_enable", True))
            or not accepted_plain
        ):
            return plain_result

        if op_local is torch.add:
            weights_cfg = getattr(lm_hp, "overlap_add_gauge_weights", [])
        elif op_local in (torch.mul, torch.multiply):
            weights_cfg = getattr(lm_hp, "overlap_mul_gauge_weights", [])
        else:
            weights_cfg = []

        gauge_weights: List[float] = []
        for w in weights_cfg or []:
            try:
                wf = float(w)
            except Exception:
                continue
            if math.isfinite(wf) and wf > 0.0:
                gauge_weights.append(wf)

        if not gauge_weights:
            return plain_result

        baseline_factories = _build_overlap_gauge_factories(weight=1.0)
        if not baseline_factories:
            print(
                "[Gauge fix] Overlap candidate has no usable gauge anchor; "
                "keeping plain accepted fit."
            )
            return plain_result

        baseline_metrics = gauge_fix_metrics(baseline_factories, raw=True)
        if not baseline_metrics:
            print(
                "[Gauge fix] Overlap candidate has no measurable gauge metric; "
                "keeping plain accepted fit."
            )
            return plain_result

        baseline_gauge_rms = max(float(m["rms"]) for m in baseline_metrics)
        print(
            "[Gauge fix] Plain fit accepted with val-loss {:.4e}; "
            "baseline raw gauge RMS={:.3e}. Testing non-zero gauge weights {}.".format(
                best_val_loss_plain,
                baseline_gauge_rms,
                [f"{w:.3g}" for w in gauge_weights],
            )
        )

        _sanitize_func_tensors(models_in[0])
        last_feasible_state = copy.deepcopy(models_in[0].state_dict())
        best_nonzero = None
        best_nonzero_state = None
        best_nonzero_weight = None
        best_nonzero_gauge_rms = None

        max_data_regress = float(getattr(lm_hp, "overlap_gauge_max_data_regress_factor", 10.0))
        improve_factor = float(getattr(lm_hp, "overlap_gauge_required_improve_factor", 0.3))
        tiny_relax = float(getattr(lm_hp, "overlap_gauge_tiny_baseline_relax_factor", 1.25))
        tiny_eps = float(getattr(lm_hp, "overlap_gauge_tiny_baseline_eps", 1.0e-10))

        for weight in gauge_weights:
            models_in[0].load_state_dict(last_feasible_state)
            gauge_factories = _build_overlap_gauge_factories(weight=weight)
            if not gauge_factories:
                print(
                    f"[Gauge fix] Could not build factories for weight={weight:.3g}; "
                    "skipping this stage."
                )
                continue

            print(f"[Gauge fix] Continuation stage: weight={weight:.3g}")
            stage_result = _train_candidate_models(
                models_in,
                accept_threshold,
                extra_train_factories=gauge_factories,
            )
            (
                _accepted_stage,
                best_val_loss_stage,
                best_train_loss_stage,
                per_val_losses_stage,
                per_train_losses_stage,
                models_in,
            ) = stage_result

            gauge_fix_diagnostic(gauge_factories, label=f"weight={weight:.3g}")
            _sanitize_func_tensors(models_in[0])
            stage_metrics = gauge_fix_metrics(gauge_factories, raw=True)
            stage_gauge_rms = max(
                (float(m["rms"]) for m in stage_metrics),
                default=float("inf"),
            )
            feasible_stage, data_cap, gauge_cap = _overlap_gauge_stage_is_feasible(
                baseline_val_loss=best_val_loss_plain,
                stage_val_loss=best_val_loss_stage,
                accept_threshold=accept_threshold,
                baseline_gauge_rms=baseline_gauge_rms,
                stage_gauge_rms=stage_gauge_rms,
                max_data_regress_factor=max_data_regress,
                required_improve_factor=improve_factor,
                tiny_baseline_relax_factor=tiny_relax,
                tiny_baseline_eps=tiny_eps,
            )
            data_ok = float(best_val_loss_stage) <= float(data_cap)
            gauge_ok = float(stage_gauge_rms) <= float(gauge_cap)
            print(
                "[Gauge fix] weight={:.3g}: val-loss {:.4e} (cap {:.4e}), "
                "raw gauge RMS {:.3e} (cap {:.3e}) -> data_ok={}, gauge_ok={}.".format(
                    weight,
                    best_val_loss_stage,
                    data_cap,
                    stage_gauge_rms,
                    gauge_cap,
                    data_ok,
                    gauge_ok,
                )
            )

            if feasible_stage:
                last_feasible_state = copy.deepcopy(models_in[0].state_dict())
                best_nonzero = (
                    True,
                    float(best_val_loss_stage),
                    float(best_train_loss_stage),
                    list(per_val_losses_stage),
                    list(per_train_losses_stage),
                    models_in,
                )
                best_nonzero_state = copy.deepcopy(last_feasible_state)
                best_nonzero_weight = float(weight)
                best_nonzero_gauge_rms = float(stage_gauge_rms)
            elif not data_ok:
                print(
                    "[Gauge fix] Data-fit cap exceeded at this weight; "
                    "stopping continuation sweep."
                )
                break

        if best_nonzero is None:
            print(
                "[Gauge fix] Rejecting overlap candidate: plain fit reached {:.4e}, "
                "but no non-zero gauge stage was feasible.".format(best_val_loss_plain)
            )
            return (
                False,
                float(best_val_loss_plain),
                float(best_train_loss_plain),
                list(per_val_losses_plain),
                list(per_train_losses_plain),
                models_in,
            )

        if best_nonzero_state is not None:
            models_in[0].load_state_dict(best_nonzero_state)
        print(
            "[Gauge fix] Accepted overlap candidate at weight={:.3g}, "
            "val-loss {:.4e}, raw gauge RMS {:.3e}.".format(
                best_nonzero_weight,
                best_nonzero[1],
                best_nonzero_gauge_rms,
            )
        )
        return best_nonzero

    def _merge_sep_candidates(cand_lists: List[List[Any]]) -> List[List[Any]]:
        """Merge per-dataset separability candidates into a shared candidate list."""
        if not cand_lists:
            return []
        if len(cand_lists) == 1:
            return cand_lists[0]
        # Key by operation + unordered group cover to be robust to group ordering.
        # Keep only candidates seen in all datasets (single shared structure).
        per_ds_maps: List[Dict[Tuple[str, frozenset, frozenset], Tuple[Any, Any, Any, Any, Any]]] = []
        for cl in cand_lists:
            m: Dict[Tuple[str, frozenset, frozenset], Tuple[Any, Any, Any, Any, Any]] = {}
            for c in cl:
                op = c[0] if len(c) > 0 else None
                g1 = c[1] if len(c) > 1 else []
                g2 = c[2] if len(c) > 2 else []
                off = c[3] if len(c) > 3 else None
                met = c[4] if len(c) > 4 else None
                op_key = "add" if op is torch.add else ("mul" if op in (torch.mul, torch.multiply) else str(op))
                k = (op_key, frozenset(g1), frozenset(g2))
                km = (op_key, frozenset(g2), frozenset(g1))
                if k not in m and km not in m:
                    m[k] = (op, list(g1), list(g2), off, met)
            per_ds_maps.append(m)

        common_keys = set(per_ds_maps[0].keys())
        for m in per_ds_maps[1:]:
            common_keys &= set(m.keys())
        merged: List[List[Any]] = []
        for k in common_keys:
            entries = [m[k] for m in per_ds_maps if k in m]
            op0, g10, g20, off0, _ = entries[0]
            mets = []
            for _, _, _, _, mi in entries:
                if mi is not None:
                    try:
                        mets.append(float(mi))
                    except Exception:
                        pass
            m_agg = _aggregate_losses(mets, mode=agg_mode, weights=(agg_weights[: len(mets)] if agg_weights else None)) if mets else None
            merged.append([op0, g10, g20, off0, m_agg])
        if any((c[4] is not None) for c in merged):
            merged.sort(
                key=lambda c: (float(c[4]) if c[4] is not None else float("inf"))
            )
        return merged

    def _eval_val_loss_no_train(model_in, val_dl):
        """Evaluate model on validation data without any training."""
        model_in.eval()
        se_sum = 0.0
        n_total = 0
        with torch.no_grad():
            for batch in val_dl:
                if not isinstance(batch, (list, tuple)) or len(batch) < 2:
                    continue
                data = tuple(t.to(device) if torch.is_tensor(t) else t for t in batch)
                r = model_in.residuals(None, data=data)
                se_sum += float((r * r).sum())
                n_total += int(r.numel())
        model_in.train()
        return se_sum / n_total if n_total > 0 else float("inf")

    def _maybe_save_stageA_good_checkpoint(reason: str) -> None:
        """Remember the latest normal-acceptable Stage-A state for safe recovery.

        Some Stage-A moves are allowed a temporary structure-first loss budget.
        If a later full refit leaves the branch under protest, recovery should
        prefer the latest normally acceptable model over the just-accepted
        degraded state.
        """
        nonlocal last_good_stageA_checkpoint, last_good_stageA_checkpoint_seq

        if model is None or current_ast is None or current_val_loss is None:
            return
        try:
            val = float(current_val_loss)
            acceptable = float(loss_acceptable_eff_init)
        except (TypeError, ValueError):
            return
        if not math.isfinite(val) or not math.isfinite(acceptable):
            return
        if val >= acceptable:
            return

        last_good_stageA_checkpoint_seq += 1
        last_good_stageA_checkpoint = {
            "seq": int(last_good_stageA_checkpoint_seq),
            "reason": str(reason),
            "x_transform_map": dict(x_transform_map),
            "train_dl": datagen_train_noshuffle,
            "val_dl": datagen_val_noshuffle,
            "current_ast": current_ast,
            "model": model,
            "models_multi": list(models_multi) if models_multi is not None else None,
            "i": int(i),
            "current_val_loss": val,
            "current_val_losses": list(current_val_losses) if current_val_losses is not None else None,
            "dual_layer_used": dual_layer_used,
            "feats": feats,
            "scale_specs": scale_specs,
            "parity_specs": parity_specs,
            "trig_spec": trig_spec,
            "invariance_feats": invariance_feats,
        }

    stageA_max_passes = max(0, int(getattr(search_hp, "stageA_max_passes", 0) or 0))
    stageA_passes_completed = 0
    stageA_pass_cap_reported = False

    def _stageA_notify_restart(reason: str) -> None:
        """Give the caller a chance to add proposals before this Stage-A state restarts."""
        if stageA_restart_callback is None:
            return
        if pending_stageA_full_refit_transaction is not None:
            return
        try:
            if current_ast is None or not collect_nn_atoms(current_ast):
                return
        except Exception:
            return
        try:
            stageA_restart_callback(
                reason=str(reason),
                current_ast=copy.deepcopy(current_ast),
                pass_index=int(stageA_passes_completed),
                y_transform_name=str(y_transform_name or y_op_str),
                x_transform_map=dict(x_transform_map or {}),
            )
        except Exception as exc:
            print(
                "[Stage A] continuation-scout callback failed: "
                f"{type(exc).__name__}: {exc}"
            )

    def _stageA_should_restart(reason: str) -> bool:
        """Return False when a bounded scout should stop at this pass boundary."""
        nonlocal stageA_passes_completed
        nonlocal stageA_pass_cap_reported

        stageA_passes_completed += 1
        if stageA_max_passes <= 0:
            _stageA_notify_restart(reason)
            return True
        if stageA_passes_completed < stageA_max_passes:
            _stageA_notify_restart(reason)
            return True
        if not stageA_pass_cap_reported:
            print(
                "[Stage A] Stage-A pass cap reached "
                f"({stageA_passes_completed}/{stageA_max_passes}) after {reason}; "
                "stopping before restart."
            )
            stageA_pass_cap_reported = True
        return False

    while separable:
        try:
            if not collect_nn_atoms(current_ast):
                print("[Stage A] No NN atoms remain; ending Stage A run.")
                separability_success = True
                break
        except Exception:
            pass

        # ---------------------------------------------------------------
        # Snapshot the last accepted model/AST. If the "full refit" fails,
        # we restore these and continue rather than regressing.
        # ---------------------------------------------------------------
        model_prev = model
        models_multi_prev = list(models_multi) if models_multi is not None else None
        ast_prev = current_ast
        dual_layer_prev = dual_layer_used
        val_loss_prev = current_val_loss
        val_losses_prev = list(current_val_losses) if current_val_losses is not None else None
        early_made_progress = False  # True when Early Compound or PureDiff changes AST this pass

        # ---------------------------------------------------------------
        # Build (or reuse) the current composite
        # ---------------------------------------------------------------
        if i == 0 and model is None:
            model0, nparam, current_ast = build_composite_ast(
                current_ast,
                model_hp.num_segments_min,
                dual_layer=False,
                leaf_builder=leaf_builder,
                device=device,
                dtype=dtype,
                reuse_leaves=reuse_leaves_init,
                freeze_non_nn=freeze_non_nn,
            )
            model0 = _apply_fit_link_to_model(model0, lm_hp)
            if is_multi:
                models_multi = [model0]
                for _di in range(1, len(train_loaders_all)):
                    m_i, _, _ = build_composite_ast(
                        current_ast,
                        model_hp.num_segments_min,
                        dual_layer=False,
                        leaf_builder=leaf_builder,
                        device=device,
                        dtype=dtype,
                        freeze_non_nn=freeze_non_nn,
                    )
                    m_i = _apply_fit_link_to_model(m_i, lm_hp)
                    models_multi.append(m_i)
                model = models_multi[0]
            else:
                model = model0
            print("Initial model (AST): {}".format(ast_to_human_readable(current_ast, x_transform_map)))

        # ===============================================================
        # Two-stage trial loop  (single-layer first, then dual-layer)
        # ===============================================================
        trial_success = False
        best_val_loss = None  # local alias; set from current_val_loss once a fit is chosen/restored
        best_train_loss_initial = None  # training chisq of initial model; used for early compound sanity check
        best_trial_loss = None
        best_trial_model = None
        best_trial_models = None
        best_trial_val_losses = None
        best_trial_ast = None
        best_trial_dual_layer = None
        best_trial_y_ok = False
        num_segments = None
        under_protest = False

        _skip_trial = skip_initial_fit and i == 0
        dual_layer_options = (True,) if search_hp.force_dual_layer else (False, True)
        if _skip_trial:
            dual_layer_options = ()  # Skip baseline training on feedback from Stage B
        for dual_layer in dual_layer_options:
            # Use num_segments_map as the target segment count, only reduce if nparam_max exceeded
            num_segments = search_hp.num_segments_map[dual_layer]
            _, model_size_est, _ = build_composite_ast(
                current_ast,
                num_segments,
                dual_layer=dual_layer,
                leaf_builder=leaf_builder,
                device=device,
                dtype=dtype,
            )
            num_segments = cap_num_segments(num_segments, num_segments, model_size_est)
            print("Setting number of segments: {}, dual_layer: {}".format(num_segments, dual_layer))

            for trial in range(search_hp.ntrial):
                # Warm-start from current model if segment counts match (same dual_layer setting)
                # This avoids random re-initialization which is fragile for multiplicative architectures
                # NOTE: We clone leaves so that if this trial is rejected, LM updates don't mutate
                # the current accepted model's parameters.
                # Build candidate model(s), warm-starting each dataset from its own previous model.
                trial_models: List[torch.nn.Module] = []
                base_models = (
                    list(models_multi) if (is_multi and models_multi is not None) else [model]
                )
                trial_ast = current_ast
                for di in range(len(train_loaders_all)):
                    base_m = base_models[di] if di < len(base_models) else model
                    reuse_leaves_di = None
                    if (
                        base_m is not None
                        and _ast_matches_arch(
                            current_ast, num_segments=num_segments, dual_layer=dual_layer
                        )
                    ):
                        reuse_map_raw_di = _build_tag_to_leaf_map(current_ast, base_m)
                        reuse_leaves_di = _clone_reuse_leaves(reuse_map_raw_di, device, dtype)
                    if di == 0:
                        trial_model_di, _, trial_ast = build_composite_ast(
                            current_ast,
                            num_segments,
                            dual_layer=dual_layer,
                            leaf_builder=leaf_builder,
                            device=device,
                            dtype=dtype,
                            reuse_leaves=reuse_leaves_di,
                            freeze_non_nn=freeze_non_nn,
                        )
                    else:
                        trial_model_di, _, _ = build_composite_ast(
                            trial_ast,
                            num_segments,
                            dual_layer=dual_layer,
                            leaf_builder=leaf_builder,
                            device=device,
                            dtype=dtype,
                            reuse_leaves=reuse_leaves_di,
                            freeze_non_nn=freeze_non_nn,
                        )
                    trial_model_di = _apply_fit_link_to_model(trial_model_di, lm_hp)
                    trial_models.append(trial_model_di)

                trial_model = trial_models[0]
                if model_hp.nparam_max is not None:
                    trial_params = trial_model.num_parameters()
                    if trial_params > model_hp.nparam_max:
                        print(
                            "Warning: model has {} parameters exceeding nparam_max {}.".format(
                                trial_params, model_hp.nparam_max
                            )
                        )
                n_leaves = len(collect_nn_atoms(current_ast))
                print(
                    f"\n{BLUE}Separability Pass:{i + 1} ({mode}), number of leaves: {n_leaves}, y-transformation: {y_op_str}(y), trial {trial + 1}/{search_hp.ntrial}, dual_layer: {dual_layer}{RESET}"
                )
                expression_human = ast_to_human_readable(current_ast, x_transform_map)
                param_count = trial_model.num_parameters()
                print(
                    "Current model: {} ({}), parameters: {}".format(
                        y_op_inv_str, expression_human, param_count
                    )
                )

                # Train initial model(s) with LM — loss_target in raw φ(y) units
                (
                    best_val_loss_trial,
                    best_train_loss_trial,
                    per_val_losses_trial,
                    _per_train_losses_trial,
                    trial_models,
                ) = _initial_fit_call_with_restart_policy(_train_initial_models, trial_models)
                trial_model = trial_models[0]

                # Extra safeguard when fit_y_link='asinh':
                # require that the model is also reasonable in (pre-link) y-space.
                trial_y_ok = True
                trial_y_mse = None
                if getattr(lm_hp, "fit_y_link", None) == "asinh":
                    try:
                        if is_multi:
                            y_mses = []
                            trial_y_ok = True
                            for di, tm in enumerate(trial_models):
                                base_model_for_sanity = (
                                    models_multi[di] if (models_multi is not None and di < len(models_multi)) else None
                                )
                                ok_i, y_mse_i, y_allow_i, D_ref_i, base_y_mse_i = _check_asinh_yspace_sanity(
                                    model=tm,
                                    dl_val=val_loaders_all[di],
                                    device=device,
                                    asinh_loss=float(per_val_losses_trial[di]),
                                    lm_hp=lm_hp,
                                    base_model=base_model_for_sanity,
                                )
                                y_mses.append(float(y_mse_i) if y_mse_i is not None else float("inf"))
                                if not ok_i:
                                    trial_y_ok = False
                                    ds_name = dataset_ids[di] if dataset_ids and di < len(dataset_ids) else f"dataset_{di}"
                                    base_str = "" if base_y_mse_i is None else f", base_y_mse={base_y_mse_i:.3e}"
                                    print(
                                        f"{RED}[Stage A] asinh y-space sanity failed ({ds_name}): "
                                        f"y-MSE={float(y_mse_i):.3e} > allowed={float(y_allow_i):.3e} "
                                        f"(asinh={float(per_val_losses_trial[di]):.3e}, D_ref={float(D_ref_i):.3e}{base_str}){RESET}"
                                    )
                            trial_y_mse = _aggregate_losses(y_mses, mode=agg_mode, weights=agg_weights)
                        else:
                            base_model_for_sanity = model  # current accepted model before this trial (may be None)
                            trial_y_ok, trial_y_mse, trial_y_allow, D_ref, base_y_mse = _check_asinh_yspace_sanity(
                                model=trial_model,
                                dl_val=datagen_val_noshuffle,
                                device=device,
                                asinh_loss=float(best_val_loss_trial),
                                lm_hp=lm_hp,
                                base_model=base_model_for_sanity,
                            )
                            if not trial_y_ok:
                                base_str = "" if base_y_mse is None else f", base_y_mse={base_y_mse:.3e}"
                                print(
                                    f"{RED}[Stage A] asinh y-space sanity failed: "
                                    f"y-MSE={float(trial_y_mse):.3e} > allowed={float(trial_y_allow):.3e} "
                                    f"(asinh={float(best_val_loss_trial):.3e}, D_ref={float(D_ref):.3e}{base_str}){RESET}"
                                )
                    except Exception as e:
                        # Don't hard-fail a trial if the sanity check itself errors
                        print(f"{YELLOW}[Stage A] Warning: asinh y-space sanity check error: {e}{RESET}")
                        trial_y_ok = True

                print(
                    "acceptable losses (raw) target={:.4e}, acceptable={:.4e} "
                    "(base units: target={:.4e}, acceptable={:.4e})".format(
                        loss_target_eff,
                        loss_acceptable_eff_init,
                        lm_hp.loss_target,
                        lm_hp.loss_acceptable,
                    )
                )

                # Track best trial even if none meet the acceptance threshold.
                # Prefer y-space-sane trials when fit_y_link='asinh' is active.
                if (
                    best_trial_loss is None
                    or (bool(trial_y_ok) and (not bool(best_trial_y_ok)))
                    or (bool(trial_y_ok) == bool(best_trial_y_ok) and best_val_loss_trial < best_trial_loss)
                ):
                    best_trial_loss = best_val_loss_trial
                    best_trial_model = trial_model
                    best_trial_models = list(trial_models)
                    best_trial_val_losses = list(per_val_losses_trial)
                    best_trial_ast = trial_ast
                    best_trial_dual_layer = dual_layer
                    best_trial_y_ok = bool(trial_y_ok)

                # Compare *raw* best_val_loss_trial against *raw* scaled threshold.
                # For fit-link branches, transformed-space acceptance can start
                # a search scaffold even when original-y sanity is not yet
                # certified.  The certificate is tracked explicitly and must be
                # earned later by visible structural progress / final validation.
                if best_val_loss_trial < loss_acceptable_eff_init:
                    candidate_note = " (as a candidate)" if mode == "quick" else ""
                    if getattr(lm_hp, "fit_y_link", None) == "asinh" and not trial_y_ok:
                        print(
                            f"{YELLOW}[Stage A] asinh trial is transformed-space acceptable "
                            "but not original-y certified; adopting as a search scaffold."
                            f"{RESET}"
                        )
                    print(
                        f"{GREEN}ADOPTING{RESET}{candidate_note} best val-loss {_loss_str(best_val_loss_trial, lm_hp)} "
                        f"(trial {trial + 1}/{search_hp.ntrial}, dual_layer={dual_layer})"
                    )
                    model = trial_model
                    models_multi = list(trial_models)
                    current_val_loss = best_val_loss_trial
                    current_val_losses = list(per_val_losses_trial)
                    best_val_loss = current_val_loss
                    best_train_loss_initial = best_train_loss_trial  # Track for early compound check
                    dual_layer_used = dual_layer
                    current_ast = trial_ast  # Keep AST in sync with model architecture
                    _refresh_fit_link_original_y_certificate("initial fit", quiet=bool(trial_y_ok))
                    trial_success = True
                    break
                else:
                    print(
                        f"Trial {trial + 1}/{search_hp.ntrial} with dual_layer={dual_layer} "
                        f"{RED}failed{RESET}. best_val_loss={best_val_loss_trial:.4e} "
                        f">= acceptable_raw={loss_acceptable_eff_init:.4e}"
                    )

            if trial_success:
                break  # stop iterating over dual_layer (False/True)

        if _skip_trial:
            # Evaluate warm-started model without training
            if is_multi and models_multi is not None:
                per_losses = [_eval_val_loss_no_train(m, vl) for m, vl in zip(models_multi, val_loaders_all)]
                current_val_loss = _aggregate_losses(per_losses, mode=agg_mode, weights=agg_weights)
                current_val_losses = per_losses
            else:
                current_val_loss = _eval_val_loss_no_train(model, datagen_val_noshuffle)
                current_val_losses = [current_val_loss]
            best_val_loss = current_val_loss
            trial_success = True
            fit_y_link_tested = True  # Block asinh reconsideration
            print(f"[skip_initial_fit] Using warm-start model, val_loss={current_val_loss:.4e}")

        # ---------------- finished two-stage attempt -------------------
        # If this was an x-preconditioning attempt, decide whether to keep it.
        if trial_success and x_precond_saved_state is not None:
            old_loss = x_precond_saved_state.get("current_val_loss", None)
            if (old_loss is not None) and (current_val_loss is not None):
                max_worsen = getattr(search_hp, "x_precond_max_worsening_factor", 100.0)
                if current_val_loss > max_worsen * old_loss:
                    kind = x_precond_saved_state.get("kind", "x-precond")
                    if trig_precond_active is not None:
                        axis, omega_tried = trig_precond_active
                        print(
                            f"[Stage A] Trig precond with omega={omega_tried:.4g} worsened val-loss "
                            f"from {old_loss:.4e} to {current_val_loss:.4e}; reverting to try next omega..."
                        )
                    else:
                        print(
                            f"[Stage A] {kind} precond worsened val-loss "
                            f"from {old_loss:.4e} to {current_val_loss:.4e}; reverting..."
                        )
                    # Restore ALL saved state (x_transform, dataloaders, AST, model, loop counter, caches)
                    (x_transform_map, datagen_train_noshuffle, datagen_val_noshuffle,
                     current_ast, model, i, current_val_loss, dual_layer_used,
                     feats, scale_specs, parity_specs, trig_spec, invariance_feats
                    ) = _restore_x_precond_state(x_precond_saved_state)
                    _refresh_units_spec_for_xmap(x_transform_map)
                    # Clear precond tracking
                    trig_precond_active = None
                    x_precond_saved_state = None
                    _x_precond_structural_gate = None
                    _x_precond_made_progress = False
                    # Reset flags to allow retry
                    under_protest = False
                    continue

            # Val-loss OK — defer final acceptance until structural progress is confirmed
            trig_precond_active = None
            _x_precond_structural_gate = x_precond_saved_state
            x_precond_saved_state = None
            _x_precond_made_progress = False

        if trial_success and pending_stageA_full_refit_transaction is not None:
            pending_stageA_full_refit_transaction = None
            _stageA_notify_restart("confirmed provisional Stage-A transaction")

        if trial_success:
            _maybe_save_stageA_good_checkpoint("post-refit")

        if not trial_success:
            print("Failed to fit model after {} trials.".format(search_hp.ntrial))

            # If we already have an accepted model from a previous pass, restore it and continue.
            if model_prev is not None:
                restored_pending_transaction = False
                restored_last_good = False
                restore_to_last_good = False
                if last_good_stageA_checkpoint is not None:
                    try:
                        prev_bad = val_loss_prev is None or float(val_loss_prev) >= float(loss_acceptable_eff_init)
                        ckpt_good = float(last_good_stageA_checkpoint["current_val_loss"]) < float(loss_acceptable_eff_init)
                        restore_to_last_good = bool(prev_bad and ckpt_good)
                    except (TypeError, ValueError, KeyError):
                        restore_to_last_good = False

                # Check if we have a validated Early Compound checkpoint that's better
                revert_to_early = (
                    early_compound_checkpoint is not None
                    and early_compound_checkpoint["val_loss"] < val_loss_prev
                )

                if pending_stageA_full_refit_transaction is not None:
                    tx = pending_stageA_full_refit_transaction
                    _failure_summary = _stageA_provisional_full_refit_failure_status(
                        candidate_loss=(
                            best_trial_loss
                            if best_trial_loss is not None
                            else tx.get("candidate_loss", val_loss_prev)
                        ),
                        parent_loss=tx.get("parent_val_loss"),
                        acceptable_loss=loss_acceptable_eff_init,
                        noise_floor_raw=float(acceptance_noise_floor_raw),
                        n_eff=_loader_size(datagen_val_noshuffle),
                    )
                    print(
                        f"{YELLOW}[Stage A] Full-refit failed after pending compound transaction "
                        f"({_failure_summary.get('status', 'failed')}); rolling back structural move(s).{RESET}"
                    )
                    for _move in list(tx.get("moves") or []):
                        _stageA_record_rejected_transaction(
                            key=tuple(_move.get("key") or ()),
                            move_kind=str(_move.get("move_kind", "compound")),
                            parent_ast=_move.get("parent_ast", tx.get("parent_ast")),
                            candidate_ast=tx.get("candidate_ast", _move.get("candidate_ast")),
                            failure_summary=_failure_summary,
                            details=_move.get("details", {}),
                            move_seq=_move.get("move_seq"),
                        )
                    x_transform_map = dict(tx.get("parent_x_transform_map") or {})
                    datagen_train_noshuffle = tx.get("parent_train_dl", datagen_train_noshuffle)
                    datagen_val_noshuffle = tx.get("parent_val_dl", datagen_val_noshuffle)
                    current_ast = tx.get("parent_ast")
                    model = tx.get("parent_model")
                    models_multi = tx.get("parent_models_multi")
                    if models_multi is None and not is_multi:
                        models_multi = [model]
                    i = int(tx.get("parent_i", i))
                    current_val_loss = tx.get("parent_val_loss")
                    current_val_losses = tx.get("parent_val_losses")
                    if current_val_losses is None and current_val_loss is not None and not is_multi:
                        current_val_losses = [current_val_loss]
                    dual_layer_used = tx.get("parent_dual_layer_used", dual_layer_used)
                    feats = tx.get("parent_feats", feats)
                    scale_specs = tx.get("parent_scale_specs", scale_specs)
                    parity_specs = tx.get("parent_parity_specs", parity_specs)
                    trig_spec = tx.get("parent_trig_spec", trig_spec)
                    invariance_feats = tx.get("parent_invariance_feats", invariance_feats)
                    _refresh_units_spec_for_xmap(x_transform_map)
                    pending_stageA_full_refit_transaction = None
                    restored_pending_transaction = True
                elif restore_to_last_good:
                    ckpt = last_good_stageA_checkpoint
                    print(
                        f"{YELLOW}[Stage A] Full-refit failed after a degraded accepted move. "
                        f"Restoring latest normal-acceptable Stage-A checkpoint "
                        f"(val_loss={float(ckpt['current_val_loss']):.4e}, "
                        f"reason={ckpt.get('reason', 'unknown')}).{RESET}"
                    )
                    x_transform_map = dict(ckpt["x_transform_map"])
                    datagen_train_noshuffle = ckpt["train_dl"]
                    datagen_val_noshuffle = ckpt["val_dl"]
                    current_ast = ckpt["current_ast"]
                    model = ckpt["model"]
                    models_multi = ckpt["models_multi"]
                    if models_multi is None and not is_multi:
                        models_multi = [model]
                    i = int(ckpt.get("i", i))
                    current_val_loss = float(ckpt["current_val_loss"])
                    current_val_losses = (
                        list(ckpt["current_val_losses"])
                        if ckpt.get("current_val_losses") is not None
                        else ([current_val_loss] if not is_multi else current_val_losses)
                    )
                    dual_layer_used = ckpt["dual_layer_used"]
                    feats = ckpt["feats"]
                    scale_specs = ckpt["scale_specs"]
                    parity_specs = ckpt["parity_specs"]
                    trig_spec = ckpt["trig_spec"]
                    invariance_feats = ckpt["invariance_feats"]
                    _refresh_units_spec_for_xmap(x_transform_map)
                    restored_last_good = True
                elif revert_to_early:
                    print(
                        f"{YELLOW}[Stage A] Full-refit failed. Reverting to validated Early Compound checkpoint "
                        f"(val_loss={early_compound_checkpoint['val_loss']:.4e}).{RESET}"
                    )
                    model = early_compound_checkpoint["model"]
                    models_multi = [model] if not is_multi else models_multi
                    current_ast = early_compound_checkpoint["ast"]
                    current_val_loss = early_compound_checkpoint["val_loss"]
                    current_val_losses = [current_val_loss] if not is_multi else current_val_losses
                    dual_layer_used = early_compound_checkpoint["dual_layer"]
                else:
                    print(
                        f"{YELLOW}[Stage A] WARNING: Full-refit failed despite warm-start. "
                        f"Restoring previous model (val_loss={val_loss_prev:.4e}). "
                        f"Consider investigating why refit degraded.{RESET}"
                    )
                    model = model_prev
                    models_multi = models_multi_prev
                    current_ast = ast_prev
                    dual_layer_used = dual_layer_prev
                    current_val_loss = val_loss_prev
                    current_val_losses = val_losses_prev

                _stageA_sync_shadow_registry(search_hp, current_ast, reason="Stage A restore")
                best_val_loss = current_val_loss
                trial_success = True
                under_protest = bool(
                    current_val_loss is not None
                    and float(current_val_loss) >= float(loss_acceptable_eff_init)
                )
                if restored_pending_transaction:
                    under_protest = False
                    print(
                        f"{YELLOW}[Stage A] Ending Stage A after rolling back the rejected compound "
                        "transaction to avoid recommitting the same scaffold."
                        f"{RESET}"
                    )
                    separable = False
                    break
                if restored_last_good:
                    under_protest = False
                    print(
                        f"{YELLOW}[Stage A] Ending Stage A after restoring the last good checkpoint "
                        "to avoid recommitting the rejected successor move."
                        f"{RESET}"
                    )
                    separable = False
                    break
                if under_protest:
                    print(
                        f"{YELLOW}[Stage A] Restored model remains under protest "
                        f"(val_loss={float(current_val_loss):.4e} >= "
                        f"acceptable={float(loss_acceptable_eff_init):.4e}); "
                        "future Stage-A moves must not regress validation loss."
                        f"{RESET}"
                    )
            else:
                # No previous accepted model (i==0 baseline): keep best trial "under protest"
                # so thresholds can be set and/or fit_only can return something usable.
                under_protest = True
                if best_trial_model is not None:
                    print(
                        f"{YELLOW}[Stage A] Keeping best baseline fit under protest: "
                        f"val_loss={best_trial_loss:.4e}, dual_layer={best_trial_dual_layer}{RESET}"
                    )
                    model = best_trial_model
                    models_multi = list(best_trial_models) if best_trial_models is not None else [best_trial_model]
                    current_ast = best_trial_ast
                    dual_layer_used = bool(best_trial_dual_layer) if best_trial_dual_layer is not None else False
                    current_val_loss = best_trial_loss
                    if best_trial_val_losses is not None:
                        current_val_losses = list(best_trial_val_losses)
                    else:
                        current_val_losses = [best_trial_loss] if not is_multi else current_val_losses
                    best_val_loss = current_val_loss
                    _refresh_fit_link_original_y_certificate(
                        "under-protest initial fit",
                        quiet=bool(best_trial_y_ok),
                    )
                else:
                    # Truly nothing usable
                    best_val_loss = None

            # In fit_only mode, accept the model "under protest" if we have one.
            # This allows threshold setting for other transforms (e.g., logneg).
            if mode == "fit_only" and model is not None:
                print(
                    f"{YELLOW}[fit_only] Accepting model 'under protest' with val_loss={best_val_loss:.4e} "
                    f"for threshold setting (need baseline fit just to set thresholds){RESET}"
                )
                # Don't break - continue to the fit_only early exit below
            elif model is not None and best_val_loss is not None:
                # Check if we can proceed "with reservations" (within 100x of threshold)
                # This applies to ALL transforms, not just identity
                reservations_factor = 100.0
                if best_val_loss < loss_acceptable_eff_init * reservations_factor:
                    # Check if we have a better saved state from before x-preconditioning
                    if x_precond_saved_state is not None:
                        old_loss = x_precond_saved_state.get("current_val_loss", None)
                        if old_loss is not None and old_loss < best_val_loss:
                            kind = x_precond_saved_state.get("kind", "x-precond")
                            print(
                                f"[Stage A] {kind} precond worsened val-loss "
                                f"from {old_loss:.4e} to {best_val_loss:.4e}; reverting..."
                            )
                            # Restore ALL saved state (including feature caches)
                            (x_transform_map, datagen_train_noshuffle, datagen_val_noshuffle,
                             current_ast, model, i, current_val_loss, dual_layer_used,
                             feats, scale_specs, parity_specs, trig_spec, invariance_feats
                            ) = _restore_x_precond_state(x_precond_saved_state)
                            _refresh_units_spec_for_xmap(x_transform_map)
                            best_val_loss = current_val_loss
                            x_precond_saved_state = None
                            _x_precond_structural_gate = None
                            _x_precond_made_progress = False
                            trig_precond_active = None  # Also clear trig-specific tracking
                            under_protest = False
                            continue

                    # No better saved state - proceed with reservations
                    print(
                        f"{YELLOW}[Stage A] Baseline fit marginal for {y_op or 'identity'}: val_loss={best_val_loss:.4e} > "
                        f"acceptable={loss_acceptable_eff_init:.4e}{RESET}"
                    )
                    print(
                        f"{YELLOW}[Stage A] Proceeding WITH RESERVATIONS to try separability search "
                        f"(within {reservations_factor:.0f}x of threshold)...{RESET}"
                    )

                    # ---------------------------------------------------------------
                    # IMMEDIATE asinh fit-link check for high dynamic range
                    # ---------------------------------------------------------------
                    auto_fit_link_threshold = getattr(lm_hp, "auto_fit_link_log_dynamic_range_threshold", 4.0)
                    if (
                        (not is_multi)
                        and y_op is None  # Only for identity transform
                        and y_log_dynamic_range is not None
                        and y_log_dynamic_range > auto_fit_link_threshold
                        and getattr(lm_hp, "fit_y_link", None) is None  # Not already using a fit-link
                    ):
                        print(
                            f"{YELLOW}Identity fit poor, and high dynamic range on y ({y_log_dynamic_range:.2f} decades > {auto_fit_link_threshold:.1f}). "
                            f"Now testing if asinh fit_y_link transformation is better than Identity{RESET}"
                        )

                        # Store original fit_y_link settings
                        orig_fit_y_link = getattr(lm_hp, "fit_y_link", None)
                        orig_fit_y_link_scale = getattr(lm_hp, "fit_y_link_scale", 1.0)

                        asinh_model, asinh_ast, asinh_val_loss, asinh_y_ok = _try_asinh_fit_with_restart_policy(
                            current_ast=current_ast,
                            num_segments=search_hp.num_segments_map[dual_layer_used],
                            dual_layer=best_trial_dual_layer or dual_layer_used or False,
                            leaf_builder=leaf_builder,
                            device=device,
                            dtype=dtype,
                            lm_hp=lm_hp,
                            y_abs_median=y_abs_median,
                            datagen_train_noshuffle=datagen_train_noshuffle,
                            datagen_val_noshuffle=datagen_val_noshuffle,
                            loss_target_eff=loss_target_eff,
                            base_model=model,
                            y_op=y_op,
                            y_op_inv=y_op_inv,
                        )

                        if asinh_model is not None and asinh_val_loss < best_val_loss:
                            _move_parent_ast = current_ast
                            _move_parent_loss = best_val_loss
                            print(f"{YELLOW}[Stage A] asinh conditioning improved fit: {best_val_loss:.4e} → {asinh_val_loss:.4e}{RESET}")
                            model = asinh_model
                            best_val_loss = asinh_val_loss
                            current_val_loss = asinh_val_loss
                            current_ast = asinh_ast
                            _stageA_record_move(
                                move_kind="fit_link_asinh_improvement",
                                parent_ast=_move_parent_ast,
                                candidate_ast=current_ast,
                                parent_loss=_move_parent_loss,
                                candidate_loss=asinh_val_loss,
                                reason="asinh fit-link improved initial Stage-A fit",
                                risk_tags={"transformed_link"},
                            )
                            _stageA_sync_shadow_registry(search_hp, current_ast, reason="asinh fit-link")
                            _refresh_fit_link_original_y_certificate(
                                "asinh fit-link",
                                quiet=bool(asinh_y_ok),
                            )
                            under_protest = bool(asinh_val_loss >= loss_acceptable_eff_init)

                            # Update acceptance threshold for subsequent candidates
                            accept_threshold_eff_cand = min(
                                accept_threshold_eff_cand,
                                max(asinh_val_loss * 1000.0, loss_target_eff * 10.0)
                            )
                            # Recompute loss_scale using asinh-space MAD
                            _, asinh_y_mad = _compute_y_med_mad_from_loader(
                                datagen_train_noshuffle,
                                device,
                                fit_y_link="asinh",
                                fit_y_link_scale=lm_hp.fit_y_link_scale,
                            )
                            if asinh_y_mad is not None and asinh_y_mad > 0:
                                loss_scale = asinh_y_mad**2
                                _refresh_effective_loss_thresholds()
                                print(
                                    f"  Updated loss_scale to asinh-space: MAD(asinh(y/scale))≈{asinh_y_mad:.3g}, "
                                    f"loss_scale={loss_scale:.3g}"
                                )

                            fit_y_link_tested = True
                            if asinh_val_loss < loss_acceptable_eff_init:
                                trial_success = True
                                print(f"{GREEN}[Stage A] asinh fit meets acceptance threshold!{RESET}")
                                print("  (NN still outputs in original y-space; model and targets are transformed only for the residual/loss calculation)")
                        else:
                            # Revert fit-link settings
                            lm_hp.fit_y_link = orig_fit_y_link
                            lm_hp.fit_y_link_scale = orig_fit_y_link_scale
                            print(f"{YELLOW}[Stage A] asinh conditioning did not improve fit ({asinh_val_loss:.4e} >= {best_val_loss:.4e}), continuing with identity{RESET}")

                    # Keep going (do NOT flip `separable` to False here).
                else:
                    # Truly hopeless fit - give up (unless this was a trig precond attempt)
                    if best_initial_loss is None:
                        model = None
                    print(
                        f"Fit too poor to proceed (val_loss={best_val_loss:.4e} > "
                        f"{reservations_factor:.0f}x acceptable={loss_acceptable_eff_init * reservations_factor:.4e})"
                    )

                    # ---------------------------------------------------------------
                    # IMMEDIATE asinh fit-link check for high dynamic range (rejected fit case)
                    # ---------------------------------------------------------------
                    auto_fit_link_threshold = getattr(lm_hp, "auto_fit_link_log_dynamic_range_threshold", 4.0)
                    if (
                        (not is_multi)
                        and y_op is None  # Only for identity transform
                        and y_log_dynamic_range is not None
                        and y_log_dynamic_range > auto_fit_link_threshold
                        and getattr(lm_hp, "fit_y_link", None) is None  # Not already using a fit-link
                    ):
                        print(
                            f"{YELLOW}Identity fit rejected, but high dynamic range on y ({y_log_dynamic_range:.2f} decades > {auto_fit_link_threshold:.1f}). "
                            f"Now testing if asinh transformation can rescue the fit{RESET}"
                        )

                        # Store original fit_y_link settings
                        orig_fit_y_link = getattr(lm_hp, "fit_y_link", None)
                        orig_fit_y_link_scale = getattr(lm_hp, "fit_y_link_scale", 1.0)

                        asinh_model, asinh_ast, asinh_val_loss, asinh_y_ok = _try_asinh_fit_with_restart_policy(
                            current_ast=current_ast,
                            num_segments=search_hp.num_segments_map[dual_layer_used],
                            dual_layer=best_trial_dual_layer or dual_layer_used or False,
                            leaf_builder=leaf_builder,
                            device=device,
                            dtype=dtype,
                            lm_hp=lm_hp,
                            y_abs_median=y_abs_median,
                            datagen_train_noshuffle=datagen_train_noshuffle,
                            datagen_val_noshuffle=datagen_val_noshuffle,
                            loss_target_eff=loss_target_eff,
                            base_model=model,
                            y_op=y_op,
                            y_op_inv=y_op_inv,
                        )

                        # Check if asinh brings fit into acceptable range (or at least "with reservations" range)
                        if asinh_model is not None and asinh_val_loss < loss_acceptable_eff_init * reservations_factor:
                            _move_parent_ast = current_ast
                            _move_parent_loss = best_val_loss
                            print(f"{GREEN}[Stage A] asinh conditioning RESCUED the fit: {best_val_loss:.4e} → {asinh_val_loss:.4e}{RESET}")
                            model = asinh_model
                            best_val_loss = asinh_val_loss
                            current_val_loss = asinh_val_loss
                            current_ast = asinh_ast
                            _stageA_record_move(
                                move_kind="fit_link_asinh_rescue",
                                parent_ast=_move_parent_ast,
                                candidate_ast=current_ast,
                                parent_loss=_move_parent_loss,
                                candidate_loss=asinh_val_loss,
                                reason="asinh fit-link rescued rejected Stage-A fit",
                                risk_tags={"transformed_link"},
                            )
                            _stageA_sync_shadow_registry(search_hp, current_ast, reason="asinh rescue")
                            _refresh_fit_link_original_y_certificate(
                                "asinh rescue",
                                quiet=bool(asinh_y_ok),
                            )
                            # Update acceptance threshold for subsequent candidates
                            accept_threshold_eff_cand = min(
                                accept_threshold_eff_cand,
                                max(asinh_val_loss * 1000.0, loss_target_eff * 10.0)
                            )
                            # Recompute loss_scale using asinh-space MAD
                            _, asinh_y_mad = _compute_y_med_mad_from_loader(
                                datagen_train_noshuffle,
                                device,
                                fit_y_link="asinh",
                                fit_y_link_scale=lm_hp.fit_y_link_scale,
                            )
                            if asinh_y_mad is not None and asinh_y_mad > 0:
                                loss_scale = asinh_y_mad**2
                                _refresh_effective_loss_thresholds()
                                print(
                                    f"  Updated loss_scale to asinh-space: MAD(asinh(y/scale))≈{asinh_y_mad:.3g}, "
                                    f"loss_scale={loss_scale:.3g}"
                                )
                            fit_y_link_tested = True
                            if asinh_val_loss < loss_acceptable_eff_init:
                                print(f"{GREEN}[Stage A] asinh fit meets acceptance threshold, continuing search{RESET}")
                            else:
                                print(f"{YELLOW}[Stage A] asinh fit marginal but within reservations range, continuing search{RESET}")
                            continue  # Re-enter the loop with the asinh-improved model
                        else:
                            # Revert fit-link settings
                            lm_hp.fit_y_link = orig_fit_y_link
                            lm_hp.fit_y_link_scale = orig_fit_y_link_scale
                            print(f"{YELLOW}[Stage A] asinh conditioning did not rescue fit ({asinh_val_loss:.4e} still too high), proceeding with rejection{RESET}")

                    if _start_initial_random_restart(
                        "Initial identity fit failed before reference save",
                        val_loss=best_val_loss,
                    ):
                        continue

                    # Check if this was an x-preconditioning attempt that failed
                    if x_precond_saved_state is not None:
                        kind = x_precond_saved_state.get("kind", "x-precond")
                        if trig_precond_active is not None:
                            axis, omega_tried = trig_precond_active
                            print(
                                f"[Stage A] Trig precond with omega={omega_tried:.4g} failed, "
                                f"reverting to try next omega candidate..."
                            )
                        else:
                            print(
                                f"[Stage A] {kind} precond failed, "
                                f"reverting to pre-precond state..."
                            )
                        # Restore ALL saved state (x_transform, dataloaders, AST, model, loop counter, caches)
                        (x_transform_map, datagen_train_noshuffle, datagen_val_noshuffle,
                         current_ast, model, i, current_val_loss, dual_layer_used,
                         feats, scale_specs, parity_specs, trig_spec, invariance_feats
                        ) = _restore_x_precond_state(x_precond_saved_state)
                        _refresh_units_spec_for_xmap(x_transform_map)
                        # Clear precond tracking
                        trig_precond_active = None
                        x_precond_saved_state = None
                        _x_precond_structural_gate = None
                        _x_precond_made_progress = False
                        # Reset flags to allow retry
                        under_protest = False
                        # DON'T break - continue loop to try next omega or other approach
                        continue

                    print("Ending the run.")
                    separable = False
                    break
            else:
                # If we never had a successful initial fit, don't return a junk model
                if best_initial_loss is None:
                    model = None
                if _start_initial_random_restart(
                    "Initial identity fit produced no usable model",
                    val_loss=best_val_loss,
                ):
                    continue
                print("Ending the run.")
                separable = False
                break

        # ---------------------------------------------------------------
        # TRIGGER 3: Opportunistic asinh when identity fit succeeds
        # but data has high dynamic range.
        # ---------------------------------------------------------------
        auto_fit_link_threshold = getattr(lm_hp, "auto_fit_link_log_dynamic_range_threshold", 4.0)
        if (
            (not is_multi)
            and trial_success
            and not under_protest
            and y_op is None
            and y_log_dynamic_range is not None
            and y_log_dynamic_range > auto_fit_link_threshold
            and getattr(lm_hp, "fit_y_link", None) is None
            and not fit_y_link_tested
        ):
            fit_y_link_tested = True
            print(
                f"{YELLOW}[Stage A] Identity fit succeeded (val_loss={best_val_loss:.4e}), "
                f"but high dynamic range ({y_log_dynamic_range:.2f} decades > {auto_fit_link_threshold:.1f}). "
                f"Opportunistically testing asinh fit-link...{RESET}"
            )
            orig_fit_y_link = getattr(lm_hp, "fit_y_link", None)
            orig_fit_y_link_scale = getattr(lm_hp, "fit_y_link_scale", 1.0)

            asinh_model, asinh_ast, asinh_val_loss, asinh_y_ok = _try_asinh_fit_with_restart_policy(
                current_ast=current_ast,
                num_segments=search_hp.num_segments_map[dual_layer_used],
                dual_layer=dual_layer_used,
                leaf_builder=leaf_builder,
                device=device,
                dtype=dtype,
                lm_hp=lm_hp,
                y_abs_median=y_abs_median,
                datagen_train_noshuffle=datagen_train_noshuffle,
                datagen_val_noshuffle=datagen_val_noshuffle,
                loss_target_eff=loss_target_eff,
                base_model=model,
                y_op=y_op,
                y_op_inv=y_op_inv,
            )

            if asinh_model is not None and asinh_val_loss < best_val_loss:
                identity_target_good, identity_target_reason = _stageA_identity_target_good(
                    val_loss=best_val_loss,
                    train_loss=best_train_loss_initial,
                    loss_target_eff=loss_target_eff,
                )
                if identity_target_good:
                    print(
                        f"{GREEN}[Stage A] Opportunistic asinh improved fit: "
                        f"{best_val_loss:.4e} → {asinh_val_loss:.4e}, "
                        f"but raw identity is target-good ({identity_target_reason}).{RESET}"
                    )
                    print(
                        "[Stage A] Retaining raw identity as the active Stage-A branch; "
                        "recording asinh only as a deferred fit-link candidate."
                    )
                    deferred_stageA_branches.append(
                        {
                            "name": "identity_asinh",
                            "base_name": "identity",
                            "fit_y_link": "asinh",
                            "fit_y_link_scale": float(getattr(lm_hp, "fit_y_link_scale", 1.0)),
                            "val_loss": float(asinh_val_loss),
                            "original_y_certified": bool(asinh_y_ok),
                            "active_val_loss": float(best_val_loss),
                            "active_train_loss": (
                                None
                                if best_train_loss_initial is None
                                else float(best_train_loss_initial)
                            ),
                            "loss_target_eff": float(loss_target_eff),
                            "reason": "identity_target_good_fitlink_deferred",
                        }
                    )
                    lm_hp.fit_y_link = orig_fit_y_link
                    lm_hp.fit_y_link_scale = orig_fit_y_link_scale
                else:
                    _move_parent_ast = current_ast
                    _move_parent_loss = best_val_loss
                    print(f"{GREEN}[Stage A] Opportunistic asinh improved fit: {best_val_loss:.4e} → {asinh_val_loss:.4e}{RESET}")
                    model = asinh_model
                    best_val_loss = asinh_val_loss
                    current_val_loss = asinh_val_loss
                    current_ast = asinh_ast
                    _stageA_record_move(
                        move_kind="fit_link_asinh_opportunistic",
                        parent_ast=_move_parent_ast,
                        candidate_ast=current_ast,
                        parent_loss=_move_parent_loss,
                        candidate_loss=asinh_val_loss,
                        reason="opportunistic asinh fit-link improved Stage-A fit",
                        risk_tags={"transformed_link"},
                    )
                    _stageA_sync_shadow_registry(search_hp, current_ast, reason="opportunistic asinh")
                    _refresh_fit_link_original_y_certificate(
                        "opportunistic asinh",
                        quiet=bool(asinh_y_ok),
                    )

                    # Recompute loss_scale using asinh-space MAD
                    _, asinh_y_mad = _compute_y_med_mad_from_loader(
                        datagen_train_noshuffle,
                        device,
                        fit_y_link="asinh",
                        fit_y_link_scale=lm_hp.fit_y_link_scale,
                    )
                    if asinh_y_mad is not None and asinh_y_mad > 0:
                        loss_scale = asinh_y_mad**2
                        _refresh_effective_loss_thresholds()
                        print(
                            f"  Updated loss_scale to asinh-space: MAD(asinh(y/scale))≈{asinh_y_mad:.3g}, "
                            f"loss_scale={loss_scale:.3g}"
                        )
            else:
                lm_hp.fit_y_link = orig_fit_y_link
                lm_hp.fit_y_link_scale = orig_fit_y_link_scale
                print(
                    f"{YELLOW}[Stage A] Opportunistic asinh did not beat identity "
                    f"({asinh_val_loss:.4e}), keeping identity{RESET}"
                )

        # Save initial loss & model only on first iteration
        if i == 0:
            _initial_fit_attempt += 1
            best_initial_loss = best_val_loss
            stageA_initial_report_val_loss = float(best_val_loss) if best_val_loss is not None else None
            stageA_initial_report_val_losses = (
                list(current_val_losses) if current_val_losses is not None else None
            )
            try:
                stageA_initial_report_n_params = int(model.num_parameters()) if model is not None else None
            except Exception:
                stageA_initial_report_n_params = None
            if trial_success:
                print("Saving initial model which converged with loss {:.6f}".format(best_initial_loss))
            else:
                # --- Retry on unlucky initialization ---
                if _start_initial_random_restart(
                    "Initial fit remained under protest",
                    val_loss=best_initial_loss,
                ):
                    continue  # re-enter while loop; i stays at 0

                print(
                    f"Saving initial model 'under protest' with loss {best_initial_loss:.6f} "
                    "(did not meet acceptance threshold)"
                )
            _refresh_fit_link_original_y_certificate("initial model save", quiet=True)
            save_dict = {
                "y_op": y_op,
                "y_op_inv": y_op_inv,
                "Nxvars": Nxvars,
                # num_segments is now stored per-atom in the AST
                "dual_layer": dual_layer_used,
                "x_transform": x_transform_map,
                "model_state_dict": model.state_dict(),
                "ast": current_ast,  # AST has correct per-atom kwargs
                "val_loss": best_val_loss,  # Track fit quality for diagnostics
                "val_losses": list(current_val_losses) if current_val_losses is not None else None,
                "val_loss_agg_mode": str(agg_mode),
                "val_loss_agg_weights": list(agg_weights) if agg_weights is not None else None,
                "dataset_ids": list(dataset_ids) if dataset_ids is not None else None,
                "fit_y_link": getattr(lm_hp, "fit_y_link", None),
                "fit_y_link_scale": float(getattr(lm_hp, "fit_y_link_scale", 1.0)),
                "fit_link_branch_certificate": dict(stageA_fit_link_certificate),
                "fit_link_branch_status": str(stageA_fit_link_certificate.get("status", "unknown")),
                "fit_link_original_y_certified": bool(
                    stageA_fit_link_certificate.get("original_y_certified", False)
                ),
                "fit_link_original_y_val_loss": stageA_fit_link_certificate.get(
                    "original_y_val_loss", None
                ),
                "fit_link_original_y_allowed_loss": stageA_fit_link_certificate.get(
                    "original_y_allowed_loss", None
                ),
                "initial_fit_random_restart": bool(_stageA_initial_restart_active),
                "initial_fit_canonical_init": (
                    False
                    if _stageA_initial_restart_active
                    else bool(getattr(lm_hp, "canonical_init", False))
                ),
                "initial_fit_evidence_enable": (
                    False
                    if _stageA_initial_restart_active
                    else bool(getattr(lm_hp, "evidence_enable", False))
                ),
                "stageA_plain_random_branch": bool(
                    _stageA_initial_restart_active or _stageA_plain_random_branch_active
                ),
            }
            torch.save(save_dict, model_output)
            if _stageA_initial_restart_active:
                _activate_plain_random_branch_after_initial_restart(
                    "Initial model was saved from the one allowed random-initialization restart"
                )
                _stageA_initial_restart_active = False

            # Reminder if asinh fit-link is active
            if getattr(lm_hp, "fit_y_link", None) == "asinh":
                asinh_scale = getattr(lm_hp, "fit_y_link_scale", 1.0)
                print(f"{YELLOW}[Note] asinh fit-link active (scale={asinh_scale:.4g}) - losses are in asinh-transformed space{RESET}")
                if not bool(stageA_fit_link_certificate.get("original_y_certified", False)):
                    print(
                        f"{YELLOW}[Note] active fit-link branch is a search scaffold; "
                        "original-y certification has not yet been earned."
                        f"{RESET}"
                    )

            # Attach best_val_loss_base (loss / loss_scale) so the caller can track
            # the global best across y-transforms for the hard_ceiling guard.
            if model is not None and current_val_loss is not None and loss_scale > 0:
                model._best_val_loss_base = current_val_loss / loss_scale

            # --- Early exit for fit_only mode ---
            # This mode is used when we just need the baseline fit to set thresholds,
            # but will do separability search with a different y-transform (e.g., quickscan winner).
            if mode == "fit_only":
                if trial_success:
                    print("[fit_only mode] Returning after initial fit without separability search.")
                else:
                    print(
                        "[fit_only mode] Returning 'under protest' model for threshold setting "
                        "(will try quickscan winner next)."
                    )
                _restore_plain_random_branch_policy("Leaving fit_only Stage A branch")
                return _units_finalize_return(False, model, None, None, candidate_sep_ops, current_ast, False, False)

        if i == 0 and not skip_initial_fit:
            # --- Discover invariance features from gradients ---------------
            invariance_feats = None  # Initialize for later use in compound detection
            try:
                print("Discovering approximate invariance features from gradients...")
                feats = discover_invariance_features(
                    model=model,
                    datagen=datagen_train_noshuffle,
                    Nxvars=Nxvars,
                    device=device,
                )
                if feats:
                    invariance_feats = feats  # Store for compound detection
                    print("Discovered {} candidate features:".format(len(feats)))
                    for fsp in feats:
                        coeff_str = ", ".join("{:+.2f}".format(c.item()) for c in fsp.coeffs)
                        print(f"  {fsp.name:15s}  kind={fsp.kind:12s}  coeffs=[{coeff_str}]")
                else:
                    print("No low-variance invariance directions found.")
            except Exception as e:
                print("Invariance discovery failed with error:", e)
            # --------------------------------------------------------------------

            # --- Discover scaling (homogeneity) features -------------------
            scale_specs = None  # Initialize for later use in compound detection
            try:
                print("Discovering approximate scaling / homogeneity features...")
                scale_specs = discover_scaling_features(
                    model=model,
                    datagen=datagen_train_noshuffle,
                    Nxvars=Nxvars,
                    device=device,
                    max_group_size=Nxvars,
                )
                if scale_specs:
                    print("Discovered {} scaling candidates:".format(len(scale_specs)))
                    for sp in scale_specs:
                        idx_str = ",".join(str(i) for i in sp.indices)
                        print(
                            f"  S=[{idx_str}]  k≈{sp.k_hat:7.3f}  "
                            f"rel_std={sp.rel_std:6.3f}  n={sp.n_points}"
                        )
                else:
                    print("No clear homogeneity detected on variable groups.")
            except Exception as e:
                print("Scaling feature discovery failed with error:", e)
            # --------------------------------------------------------------------

            # --- Oracle-based scaling probe verification --------------------
            oracle_specs = None
            if scale_specs:
                try:
                    # Extract compound targets from current AST for oracle probe
                    compound_targets_oracle = _extract_compound_targets_from_ast(current_ast)
                    if compound_targets_oracle:
                        print(f"[Oracle Probe] Found {len(compound_targets_oracle)} compound target(s) from AST")

                    oracle_specs = probe_oracle_scaling(
                        model=model,
                        datagen=datagen_train_noshuffle,
                        Nxvars=Nxvars,
                        device=device,
                        rel_std_threshold=float(getattr(search_hp, "oracle_scaling_rel_std", 0.08)),
                        compound_targets=compound_targets_oracle,
                        gradient_specs=scale_specs,
                    )
                    if oracle_specs:
                        print(f"[Oracle Probe] Verified {len(oracle_specs)} scaling spec(s):")
                        for osp in oracle_specs:
                            # Show compound_name if present, otherwise show indices
                            if osp.compound_name:
                                display = osp.compound_name
                            else:
                                display = ",".join(str(i) for i in osp.indices)
                            print(
                                f"  S=[{display}]  k≈{osp.oracle_k:7.3f}  "
                                f"rel_std={osp.oracle_rel_std:6.3f}  n={osp.n_points}"
                            )
                        # Merge direct certificates back by the complete raw-axis
                        # group, including joint-only homogeneity proposals.
                        oracle_by_group = {
                            tuple(int(i) for i in sp.indices): sp
                            for sp in oracle_specs
                            if sp.indices and not sp.compound_name
                        }
                        for sp in scale_specs:
                            group = tuple(int(i) for i in sp.indices)
                            osp = oracle_by_group.get(group)
                            if osp is None:
                                continue
                            sp.oracle_verified = True
                            sp.oracle_k = osp.oracle_k
                            sp.oracle_rel_std = osp.oracle_rel_std
                    else:
                        print("[Oracle Probe] No scaling verified by direct evaluation.")
                except Exception as e:
                    print(f"[Oracle Probe] Failed with error: {e}")

            # --- Trig z-scaling probe: monomial degree in cos/sin ----------
            trig_scale_specs = []
            trig_axis_specs_all = []
            try:
                # Extract compound targets from current AST (may have compounds from previous iterations)
                compound_targets_stageA = _extract_compound_targets_from_ast(current_ast)
                if compound_targets_stageA:
                    print(f"  [Trig Scaling] Found {len(compound_targets_stageA)} compound target(s) from AST")

                trig_scale_specs = probe_trig_scaling(
                    model=model,
                    datagen=datagen_train_noshuffle,
                    Nxvars=Nxvars,
                    device=device,
                    oracle_specs=oracle_specs,
                    compound_targets=compound_targets_stageA,
                )
                if trig_scale_specs:
                    for ts in trig_scale_specs:
                        display = ts.compound_name if ts.compound_name else f"x{ts.axis}"
                        print(
                            f"  [Trig Scaling] {display}: "
                            f"{ts.trig_fn}({ts.omega:.4g}·x)^{ts.k_hat:.2f}  "
                            f"rel_std={ts.rel_std:.3f}  n={ts.n_points}"
                        )
                else:
                    print("  [Trig Scaling] No monomial-in-trig detected.")
            except Exception as e:
                print(f"  [Trig Scaling] Failed: {e}")

            # Derive trig axis specs from oracle (primary source for all downstream logic)
            import math as _math  # local import to satisfy ruff F823
            for ts in trig_scale_specs:
                trig_axis_specs_all.append(TrigAxisSpec(
                    axis=ts.axis,
                    omega=ts.omega,
                    strength=100.0,  # synthetic; oracle-verified
                    n_points=ts.n_points,
                    tmin=0.0, tmax=0.0,
                    phase=0.0 if ts.trig_fn == "cos" else _math.pi / 2,
                    basis_fn=str(getattr(ts, "basis_fn", "") or getattr(ts, "trig_fn", "")),
                ))
            # --------------------------------------------------------------------

            # --- Protected GS preflight -----------------------------------------
            # Legacy early-scaling and Early PureDiff passes predate the shared
            # GS carrier bank.  Give one certified full-support GS coordinate an
            # empirical trial on the untouched root atom before either legacy
            # pass may commit.  Failure leaves both legacy paths unchanged.
            gs_preflight_accepted = False
            if (not is_multi) and mode != "fit_only":
                try:
                    gs_atoms = collect_nn_atoms(current_ast)
                    if (
                        len(gs_atoms) == 1
                        and set(gs_atoms[0].var_idxs) == set(range(Nxvars))
                        and not has_nontrivial_input(gs_atoms[0])
                    ):
                        gs_atom = gs_atoms[0]
                        gs_tag_to_leaf = _build_tag_to_leaf_map(current_ast, model)
                        (
                            gs_accepted,
                            gs_model,
                            gs_ast,
                            gs_loss,
                            gs_full,
                            _gs_enables_sep,
                        ) = _try_stageA_decisive_gs_preflight_for_atom(
                            model=model,
                            current_ast=current_ast,
                            atom=gs_atom,
                            tag_to_leaf=gs_tag_to_leaf,
                            datagen_train_noshuffle=datagen_train_noshuffle,
                            datagen_val_noshuffle=datagen_val_noshuffle,
                            device=device,
                            dtype=dtype,
                            leaf_builder=leaf_builder,
                            dual_layer_used=dual_layer_used,
                            search_hp=search_hp,
                            lm_hp=lm_hp,
                            loss_target_eff=loss_target_eff,
                            accept_threshold_eff_cand=loss_acceptable_eff_init,
                            best_val_loss=best_val_loss,
                            current_val_loss=current_val_loss,
                            stageA_under_protest=bool(under_protest),
                            best_train_loss=best_train_loss_initial,
                            loss_scale=loss_scale,
                            model_sep_output=model_sep_output,
                            y_op=y_op,
                            y_op_inv=y_op_inv,
                            Nxvars=Nxvars,
                            x_transform_map=x_transform_map,
                            trig_spec=trig_spec,
                            units_spec=units_spec,
                            enforce_units=bool(enforce_units),
                            units_reject_cb=_units_reject,
                            scaling_features=scale_specs,
                        )
                        if gs_accepted:
                            parent_ast = current_ast
                            parent_loss = best_val_loss
                            if ast_to_human_readable(gs_ast) == ast_to_human_readable(parent_ast):
                                print(
                                    "[Stage A GS Preflight] Accepted candidate kept AST "
                                    "unchanged; ignoring no-op accept."
                                )
                            else:
                                # Commit first.  Everything after this boundary is
                                # best-effort bookkeeping and cannot reopen the
                                # legacy fallback on a partially updated state.
                                model = gs_model
                                current_ast = gs_ast
                                best_val_loss = gs_loss
                                current_val_loss = gs_loss
                                separability_success = True
                                early_made_progress = True
                                gs_preflight_accepted = True
                                if gs_full:
                                    full_compound_solved = True
                                _stageA_record_decisive_gs_preflight_best_effort(
                                    candidate_model=gs_model,
                                    parent_ast=parent_ast,
                                    candidate_ast=current_ast,
                                    parent_loss=parent_loss,
                                    candidate_loss=gs_loss,
                                    full_compound=bool(gs_full),
                                    search_hp=search_hp,
                                    move_details=_stageA_compound_move_details,
                                    record_move=_stageA_record_move,
                                    sync_shadow=_stageA_sync_shadow_registry,
                                )
                                if gs_full:
                                    print(
                                        "[Stage A GS Preflight] Full-variable carrier "
                                        "compressed; outer map still unresolved."
                                    )
                                print(
                                    f"{GREEN}[Stage A GS Preflight] Accepted; "
                                    f"val-loss {float(gs_loss):.4e}.{RESET}"
                                )
                except Exception as exc:
                    if gs_preflight_accepted:
                        print(
                            "[Stage A GS Preflight] Post-commit diagnostic failed; "
                            "keeping the accepted GS state: "
                            f"{type(exc).__name__}: {exc}"
                        )
                    else:
                        print(
                            "[Stage A GS Preflight] Failed safely; continuing with "
                            f"legacy early compounds: {type(exc).__name__}: {exc}"
                        )

            # --- Early compound detection from scaling exponents -------------
            # If single-variable scalings have clean integer exponents with low
            # scatter, try a compound variable z = Π xᵢ^kᵢ early.
            # This is an opportunity for quick progress, not an escape hatch -
            # feature discovery continues regardless of result.
            if (
                not gs_preflight_accepted
                and (not is_multi)
                and scale_specs
                and mode != "fit_only"
            ):
                try:
                    early_candidates = _check_early_compound_from_scaling(
                        scale_specs,
                        Nxvars,
                        rel_std_threshold=float(getattr(search_hp, "early_compound_rel_std", 0.05)),
                        k_int_threshold=float(getattr(search_hp, "early_compound_k_int", 0.15)),
                        require_oracle=bool(oracle_specs),
                        soft_noise_floor_raw=float(acceptance_noise_floor_raw),
                        search_hp=search_hp,
                    )
                    if early_candidates:
                        print(f"[Early Compound] Found {len(early_candidates)} candidate(s), trying largest first...")
                        early_soft_keys = set()
                        try:
                            for _prop in _stageA_noisy_soft_monomial_product_proposals_from_scaling(
                                scale_specs,
                                tuple(range(int(Nxvars))),
                                search_hp=search_hp,
                                noise_floor_raw=float(acceptance_noise_floor_raw),
                            ):
                                pattern = tuple(int(v) for v in _prop[0])
                                _z_idxs = tuple(i for i, exp in enumerate(pattern) if int(exp) != 0)
                                if len(_z_idxs) < 2:
                                    continue
                                _z_exps = tuple(int(pattern[i]) for i in _z_idxs)
                                early_soft_keys.add((_z_idxs, _z_exps))
                        except Exception:
                            early_soft_keys = set()
                        early_candidate_meta = {}

                        # --- Oracle null-test filter ---
                        filtered = []
                        for z_var_idxs, z_exponents, remaining in early_candidates:
                            _ec_key = (tuple(int(v) for v in z_var_idxs), tuple(int(v) for v in z_exponents))
                            _ec_meta = {
                                "soft_monomial_compound": bool(_ec_key in early_soft_keys),
                                "null_verified": None,
                            }
                            if len(z_var_idxs) < 2:
                                early_candidate_meta[_ec_key] = _ec_meta
                                filtered.append((z_var_idxs, z_exponents, remaining))
                                continue
                            try:
                                null_res = verify_compound_null_test(
                                    model=model,
                                    datagen=datagen_train_noshuffle,
                                    z_var_idxs=z_var_idxs,
                                    z_exponents=z_exponents,
                                    Nxvars=Nxvars,
                                    device=device,
                                )
                                idx_str = ",".join(str(i) for i in z_var_idxs)
                                exp_str = ",".join(str(e) for e in z_exponents)
                                if null_res.verified:
                                    print(f"  [Null Test] z=[{idx_str}]^[{exp_str}] VERIFIED "
                                          f"(median_dev={null_res.median_dev:.4f}, n={null_res.n_valid})")
                                    _ec_meta["null_verified"] = True
                                    filtered.append((z_var_idxs, z_exponents, remaining))
                                else:
                                    print(f"  [Null Test] z=[{idx_str}]^[{exp_str}] not verified "
                                          f"(median_dev={null_res.median_dev:.4f}, n={null_res.n_valid}) — keeping anyway")
                                    _ec_meta["null_verified"] = False
                                    filtered.append((z_var_idxs, z_exponents, remaining))
                            except Exception as e:
                                print(f"  [Null Test] Error: {e} — keeping candidate")
                                _ec_meta["null_verified"] = False
                                filtered.append((z_var_idxs, z_exponents, remaining))
                            early_candidate_meta[_ec_key] = _ec_meta
                        early_candidates = filtered

                        # Get NN atoms - only proceed if single atom covering all variables
                        nn_atoms = collect_nn_atoms(current_ast)
                        if len(nn_atoms) == 1 and set(nn_atoms[0].var_idxs) == set(range(Nxvars)):
                            atom = nn_atoms[0]
                            tag_to_leaf = _build_tag_to_leaf_map(current_ast, model)

                            # --- Trig-extended compound candidates from trig z-scaling probe ---
                            # Use trig_scale_specs from the self-contained ω probe (computed above).
                            # These are reliable — they come from direct scaling tests, not FFT.
                            early_trig_specs = trig_scale_specs  # List[TrigScaleSpec]

                            for z_var_idxs, z_exponents, remaining_var_idxs in early_candidates:
                                early_key = (tuple(int(v) for v in z_var_idxs), tuple(int(v) for v in z_exponents))
                                early_meta = early_candidate_meta.get(early_key, {})
                                early_soft_candidate = bool(
                                    early_meta.get("soft_monomial_compound", False)
                                    and float(acceptance_noise_floor_raw) > 0.0
                                )
                                early_provisional_candidate = bool(
                                    early_soft_candidate
                                    or (
                                        early_meta.get("null_verified", None) is False
                                        and float(acceptance_noise_floor_raw) > 0.0
                                    )
                                )
                                pre_ec_model = model
                                pre_ec_ast = current_ast
                                pre_ec_best_val_loss = best_val_loss
                                pre_ec_current_val_loss = current_val_loss
                                pre_ec_separability_success = separability_success
                                pre_ec_early_made_progress = early_made_progress
                                early_transaction_key = _stageA_early_compound_transaction_key(
                                    parent_ast=pre_ec_ast,
                                    z_var_idxs=z_var_idxs,
                                    z_exponents=z_exponents,
                                    remaining_var_idxs=remaining_var_idxs,
                                )
                                if early_transaction_key in stageA_rejected_transactions:
                                    print(
                                        "[Early Compound] Skipping previously rejected provisional transaction: "
                                        f"z vars={z_var_idxs}, exponents={z_exponents}, remaining={remaining_var_idxs}"
                                    )
                                    continue
                                print(f"[Early Compound] Trying z vars: {z_var_idxs}, exponents: {z_exponents}")
                                if remaining_var_idxs:
                                    print(f"[Early Compound] Remaining vars: {remaining_var_idxs}")

                                # --- Try trig-extended candidate first when remaining vars have trig ---
                                # If remaining vars show strong trig behavior, try trig-extended
                                # compound BEFORE the multiplicative split. For
                                # y=x0*x1*sin(x2), we want NN[z=x0*x1*sin(x2)]
                                # rather than NN[z=x0*x1] * NN[x2].
                                trig_extended_tried = False
                                if remaining_var_idxs and early_trig_specs:
                                    for rem_idx in remaining_var_idxs:
                                        trig_spec = next(
                                            (s for s in early_trig_specs if int(s.axis) == int(rem_idx)),
                                            None
                                        )
                                        if trig_spec is not None and float(trig_spec.rel_std) < 0.10:
                                            # Build trig-extended z_ast: z * sin(ω*rem_idx) or z * cos(ω*rem_idx)
                                            omega = snap_omega(float(trig_spec.omega))
                                            trig_kind = trig_spec.trig_fn

                                            base_z_ast = build_monomial_ast(z_var_idxs, z_exponents)
                                            omega_node = MulNode(ConstNode(float(omega)), Var(int(rem_idx)))
                                            trig_node = SinNode(omega_node) if trig_kind == "sin" else CosNode(omega_node)
                                            extended_z_ast = MulNode(base_z_ast, trig_node)

                                            units_reason = _analytic_units_rejection(
                                                extended_z_ast,
                                                units_spec,
                                                enforce_units=bool(enforce_units),
                                            )
                                            if units_reason is not None:
                                                print(
                                                    f"[Units] Skipping Early Compound trig-extended "
                                                    f"z * {trig_kind}({omega:.3g}*x{rem_idx}): {units_reason}"
                                                )
                                                _units_reject("early_compound_trig", units_reason)
                                                continue

                                            # Extended z vars = original z vars + trig var
                                            extended_z_var_idxs = tuple(sorted(set(z_var_idxs) | {rem_idx}))
                                            # Remaining = original remaining minus absorbed trig var
                                            new_remaining = tuple(r for r in remaining_var_idxs if r != rem_idx)

                                            print(f"[Early Compound] Trying trig-extended: z * {trig_kind}({omega:.3g}*x{rem_idx})")

                                            ec_ext_accepted, ec_ext_model, ec_ext_ast, ec_ext_loss = _try_early_compound_candidate(
                                                z_var_idxs=extended_z_var_idxs,
                                                z_exponents=None,  # Not used when z_ast_override is provided
                                                remaining_var_idxs=new_remaining,
                                                model=model,
                                                current_ast=current_ast,
                                                atom=atom,
                                                tag_to_leaf=tag_to_leaf,
                                                datagen_train_noshuffle=datagen_train_noshuffle,
                                                datagen_val_noshuffle=datagen_val_noshuffle,
                                                device=device,
                                                dtype=dtype,
                                                leaf_builder=leaf_builder,
                                                dual_layer_used=dual_layer_used,
                                                search_hp=search_hp,
                                                lm_hp=lm_hp,
                                                loss_target_eff=loss_target_eff,
                                                accept_threshold_eff=loss_acceptable_eff_init,
                                                best_val_loss=best_val_loss,
                                                best_train_loss_initial=best_train_loss_initial,
                                                loss_scale=loss_scale,
                                                model_sep_output=model_sep_output,
                                                y_op=y_op,
                                                y_op_inv=y_op_inv,
                                                Nxvars=Nxvars,
                                                x_transform_map=x_transform_map,
                                                z_ast_override=extended_z_ast,
                                            )

                                            if ec_ext_accepted:
                                                _move_parent_ast = current_ast
                                                _move_parent_loss = best_val_loss
                                                model = ec_ext_model
                                                current_ast = ec_ext_ast
                                                best_val_loss = ec_ext_loss
                                                current_val_loss = ec_ext_loss
                                                if not early_provisional_candidate:
                                                    _stageA_record_move(
                                                        move_kind="early_compound_trig_extension",
                                                        parent_ast=_move_parent_ast,
                                                        candidate_ast=current_ast,
                                                        parent_loss=_move_parent_loss,
                                                        candidate_loss=ec_ext_loss,
                                                        reason="early compound trig extension accepted",
                                                        risk_tags={"compound_coordinate"},
                                                    )
                                                separability_success = True
                                                trig_extended_tried = True
                                                print(f"{GREEN}[Early Compound] Trig-extended ACCEPTED!{RESET}")
                                                break  # Stop trying trig vars for this candidate

                                if trig_extended_tried:
                                    # Trig-extended succeeded - skip normal multiplicative split
                                    # but still run full validation pass below
                                    ec_accepted = True
                                    ec_model = model
                                    ec_ast = current_ast
                                    ec_loss = best_val_loss
                                else:
                                    # --- Normal multiplicative split (original behavior) ---
                                    ec_accepted, ec_model, ec_ast, ec_loss = _try_early_compound_candidate(
                                        z_var_idxs=z_var_idxs,
                                        z_exponents=z_exponents,
                                        remaining_var_idxs=remaining_var_idxs,
                                        model=model,
                                        current_ast=current_ast,
                                        atom=atom,
                                        tag_to_leaf=tag_to_leaf,
                                        datagen_train_noshuffle=datagen_train_noshuffle,
                                        datagen_val_noshuffle=datagen_val_noshuffle,
                                        device=device,
                                        dtype=dtype,
                                        leaf_builder=leaf_builder,
                                        dual_layer_used=dual_layer_used,
                                        search_hp=search_hp,
                                        lm_hp=lm_hp,
                                        loss_target_eff=loss_target_eff,
                                        accept_threshold_eff=loss_acceptable_eff_init,
                                        best_val_loss=best_val_loss,
                                        best_train_loss_initial=best_train_loss_initial,
                                        loss_scale=loss_scale,
                                        model_sep_output=model_sep_output,
                                        y_op=y_op,
                                        y_op_inv=y_op_inv,
                                        Nxvars=Nxvars,
                                        x_transform_map=x_transform_map,
                                    )

                                if ec_accepted:
                                    _move_parent_ast = current_ast
                                    _move_parent_loss = best_val_loss
                                    model = ec_model
                                    current_ast = ec_ast
                                    best_val_loss = ec_loss
                                    current_val_loss = ec_loss
                                    if not trig_extended_tried and not early_provisional_candidate:
                                        _stageA_record_move(
                                            move_kind="early_compound",
                                            parent_ast=_move_parent_ast,
                                            candidate_ast=current_ast,
                                            parent_loss=_move_parent_loss,
                                            candidate_loss=ec_loss,
                                            reason="early compound candidate accepted",
                                            risk_tags={"compound_coordinate"},
                                        )
                                    if not early_provisional_candidate:
                                        _stageA_sync_shadow_registry(search_hp, current_ast, reason="early compound")
                                    separability_success = True
                                    print("[Early Compound] Accepted! Running full validation pass...")
                                    early_made_progress = True

                                    # --- Full validation pass for Early Compound ---
                                    # Build composite with standard segments (same as main trial loop)
                                    ec_num_segments = search_hp.num_segments_map.get(dual_layer_used, 16)
                                    ec_trial_model, _, ec_trial_ast = build_composite_ast(
                                        current_ast,
                                        ec_num_segments,
                                        dual_layer=dual_layer_used,
                                        leaf_builder=leaf_builder,
                                        device=device,
                                        dtype=dtype,
                                    )
                                    ec_trial_model = _apply_fit_link_to_model(ec_trial_model, lm_hp)

                                    print(f"[Early Compound] Full validation: segments={ec_num_segments}, dual_layer={dual_layer_used}")

                                    ec_val_loss, _, ec_val_params, ec_lm_opt = fit_initial_model_with_tournament(
                                        ec_trial_model,
                                        datagen_train_noshuffle,
                                        datagen_val_noshuffle,
                                        epochs=lm_hp.epochs,
                                        LM_strategy=lm_hp.strategy,
                                        nval_patience=lm_hp.nval_patience,
                                        loss_target=loss_target_eff,
                                        epochs_min=lm_hp.epochs_min,
                                        chisq_tol=lm_hp.chisq_tol,
                                        device=device,
                                        epochs_awful_check=lm_hp.epochs_awful_check,
                                        awful_threshold=lm_hp.awful_threshold,
                                        lm_verbose=lm_hp.LM_verbose,
                                        y_op=y_op,
                                        y_op_inv=y_op_inv,
                                        lm_hp=lm_hp,
                                    )
                                    ec_lm_opt._update_param_groups(ec_val_params)

                                    if ec_val_loss < loss_acceptable_eff_init:
                                        print(f"{GREEN}[Early Compound] Full validation PASSED{RESET}: val_loss={ec_val_loss:.4e} < threshold={loss_acceptable_eff_init:.4e}")
                                        _move_parent_ast = pre_ec_ast if early_provisional_candidate else current_ast
                                        _move_parent_loss = pre_ec_best_val_loss if early_provisional_candidate else best_val_loss
                                        model = ec_trial_model
                                        current_ast = ec_trial_ast
                                        best_val_loss = ec_val_loss
                                        current_val_loss = ec_val_loss
                                        _early_validation_tags = {"compound_coordinate"}
                                        if bool(early_meta.get("soft_monomial_compound", False)):
                                            _early_validation_tags.add("soft_monomial_compound")
                                        if early_meta.get("null_verified", None) is False:
                                            _early_validation_tags.add("null_unverified")
                                        _stageA_record_move(
                                            move_kind="early_compound_validation",
                                            parent_ast=_move_parent_ast,
                                            candidate_ast=current_ast,
                                            parent_loss=_move_parent_loss,
                                            candidate_loss=ec_val_loss,
                                            reason=(
                                                "early provisional compound full validation passed"
                                                if early_provisional_candidate
                                                else "early compound full validation passed"
                                            ),
                                            risk_tags=_early_validation_tags,
                                            details={
                                                "full_refit_confirmed": True,
                                                "soft_monomial_compound": bool(
                                                    early_meta.get("soft_monomial_compound", False)
                                                ),
                                                "null_verified": early_meta.get("null_verified", None),
                                                "z_var_idxs": [int(v) for v in z_var_idxs],
                                                "z_exponents": [int(v) for v in z_exponents],
                                                "remaining_var_idxs": [int(v) for v in remaining_var_idxs],
                                            },
                                        )
                                        _stageA_sync_shadow_registry(search_hp, current_ast, reason="early compound validation")

                                        # Save validated checkpoint for potential revert
                                        early_compound_checkpoint = {
                                            "model": model,
                                            "ast": current_ast,
                                            "val_loss": ec_val_loss,
                                            "dual_layer": dual_layer_used,
                                        }
                                    else:
                                        print(f"{YELLOW}[Early Compound] Full validation FAILED{RESET}: val_loss={ec_val_loss:.4e} >= threshold={loss_acceptable_eff_init:.4e}")
                                        if early_provisional_candidate:
                                            _failure_summary = _stageA_provisional_full_refit_failure_status(
                                                candidate_loss=ec_val_loss,
                                                parent_loss=pre_ec_best_val_loss,
                                                acceptable_loss=loss_acceptable_eff_init,
                                                noise_floor_raw=float(acceptance_noise_floor_raw),
                                                n_eff=_loader_size(datagen_val_noshuffle),
                                            )
                                            print(
                                                "[Early Compound] Provisional compound failed full validation "
                                                f"({_failure_summary.get('status', 'failed')}); rolling back "
                                                "the structural move."
                                            )
                                            _stageA_record_rejected_transaction(
                                                key=early_transaction_key,
                                                move_kind="early_compound_validation",
                                                parent_ast=pre_ec_ast,
                                                candidate_ast=ec_trial_ast,
                                                failure_summary=_failure_summary,
                                                details={
                                                    "soft_monomial_compound": bool(
                                                        early_meta.get("soft_monomial_compound", False)
                                                    ),
                                                    "null_verified": early_meta.get("null_verified", None),
                                                    "z_var_idxs": [int(v) for v in z_var_idxs],
                                                    "z_exponents": [int(v) for v in z_exponents],
                                                    "remaining_var_idxs": [int(v) for v in remaining_var_idxs],
                                                },
                                            )
                                            model = pre_ec_model
                                            current_ast = pre_ec_ast
                                            best_val_loss = pre_ec_best_val_loss
                                            current_val_loss = pre_ec_current_val_loss
                                            separability_success = pre_ec_separability_success
                                            early_made_progress = pre_ec_early_made_progress
                                            early_compound_checkpoint = None
                                            continue
                                        print("[Early Compound] Keeping original acceptance but NOT saving as checkpoint")
                                        early_compound_checkpoint = None

                                    if (y_op is None) and _is_pure_1d_full_compound_ast(current_ast, Nxvars):
                                        if not full_compound_solved:
                                            print(
                                                "[Early Compound] Full-variable 1D compound compressed; "
                                                "outer map still unresolved, y-search remains open"
                                            )
                                        full_compound_solved = True

                                    break  # Stop on first success
                                # Only try the best (first) candidate - if it fails, smaller
                                # subsets are unlikely to help given candidates are sorted by size
                                break
                    else:
                        # Diagnostic: show why early compound didn't find anything
                        # Use the same shared helper so the diagnostic is consistent
                        single_var_specs_diag = [sp for sp in scale_specs if len(sp.indices) == 1]
                        if single_var_specs_diag:
                            rel_std_thresh = float(getattr(search_hp, "early_compound_rel_std", 0.05))
                            k_int_thresh = float(getattr(search_hp, "early_compound_k_int", 0.15))
                            qualifying_diag = _get_qualifying_scaling_vars(
                                scale_specs,
                                rel_std_threshold=rel_std_thresh,
                                k_int_threshold=k_int_thresh,
                            )
                            print(f"[Early Compound] Checked {len(single_var_specs_diag)} single-var scaling specs, "
                                  f"{len(qualifying_diag)} qualify (need ≥2 for compound)")
                except Exception as e:
                    print(f"[Early Compound] Detection failed: {e}")
            # --------------------------------------------------------------------

            # --- Early power-difference detection (gradient-ratio scan) ----------
            # Detects z = xi^n - xj^n for all integer n (including n=1 linear diff).
            # The gradient-ratio scan is cheap (just ratios of gradients we already have),
            # so it always runs on single-atom models with ≥2 variables.
            # This covers structures such as cos(x4-x5) and (x1²-x3²)².
            if (
                not gs_preflight_accepted
                and (not is_multi)
                and mode != "fit_only"
            ):
                try:
                    nn_atoms = collect_nn_atoms(current_ast)
                    # Only proceed if single atom covering all variables
                    if len(nn_atoms) == 1 and set(nn_atoms[0].var_idxs) == set(range(Nxvars)):
                        atom = nn_atoms[0]
                        if len(atom.var_idxs) >= 2:
                            tag_to_leaf = _build_tag_to_leaf_map(current_ast, model)
                            leaf = tag_to_leaf.get(atom.tag)

                            if leaf is not None:
                                import numpy as np

                                # Use eval_inputs (works for both simple and compound atoms)
                                x_list, dydx_list = [], []
                                n_batches = 0
                                max_batches = 4
                                for batch in datagen_train_noshuffle:
                                    if isinstance(batch, (list, tuple)):
                                        x, _ = batch
                                    else:
                                        x = batch
                                    x = x.to(device)
                                    with torch.no_grad():
                                        x_sub, _, _ = eval_inputs(atom, x, need_grad=False, need_hess=False)
                                        cache = {"x": x_sub}
                                        g = leaf.grad(cache)
                                        if g.dim() == 2:
                                            g = g.unsqueeze(1)
                                        g = g[:, 0, :]
                                    x_list.append(x_sub.detach().cpu().numpy())
                                    dydx_list.append(g.detach().cpu().numpy())
                                    n_batches += 1
                                    if n_batches >= max_batches:
                                        break

                                cols, z_ast_existing_early = _atom_compound_cols(atom)
                                atom_is_compound = z_ast_existing_early is not None
                                early_already_sep = False
                                if atom_is_compound:
                                    try:
                                        inputs_early = get_input_exprs(atom)
                                        z_expr_early = inputs_early[0]
                                        extra_var_idxs_early = [
                                            int(inp.var_idxs[0])
                                            for inp in inputs_early[1:]
                                            if is_trivial_input(inp)
                                        ]
                                        extra_input_asts_early = [
                                            inp for inp in inputs_early[1:]
                                            if not is_trivial_input(inp)
                                        ]
                                        if extra_var_idxs_early or extra_input_asts_early:
                                            early_already_sep = _quick_separability_check(
                                                model=model,
                                                leaf=leaf,
                                                z_expr=z_expr_early,
                                                extra_var_idxs=extra_var_idxs_early,
                                                extra_input_asts=extra_input_asts_early,
                                                datagen_train=datagen_train_noshuffle,
                                                device=device,
                                                dtype=dtype,
                                            )
                                    except Exception:
                                        early_already_sep = False
                                if x_list:
                                    x_vals = np.concatenate(x_list, axis=0)
                                    dydx_vals = np.concatenate(dydx_list, axis=0)

                                    # Call power-difference detection (gradient-ratio primary,
                                    # invariance features as fallback for n=1)
                                    pure_diff_proposals = _detect_pure_difference_compounds(
                                        x_vals=x_vals,
                                        dydx_vals=dydx_vals,
                                        var_idxs=tuple(cols),
                                        invariance_feats=invariance_feats if invariance_feats else None,
                                        precision=float(getattr(search_hp, "compound_threshold", 0.05)),
                                        z_ast_existing=z_ast_existing_early,
                                        units_spec=units_spec,
                                        enforce_units=bool(enforce_units),
                                    )

                                    if pure_diff_proposals:
                                        print(f"[Early PureDiff] Found {len(pure_diff_proposals)} power-difference proposals, trying...")
                                        pd_accepted, pd_model, pd_ast, pd_loss, pd_full, _pd_enables_sep = _try_compound_candidates_for_atom(
                                            proposals=pure_diff_proposals,
                                            model=model,
                                            current_ast=current_ast,
                                            atom=atom,
                                            tag_to_leaf=tag_to_leaf,
                                            datagen_train_noshuffle=datagen_train_noshuffle,
                                            datagen_val_noshuffle=datagen_val_noshuffle,
                                            device=device,
                                            dtype=dtype,
                                            leaf_builder=leaf_builder,
                                            dual_layer_used=dual_layer_used,
                                            search_hp=search_hp,
                                            lm_hp=lm_hp,
                                            loss_target_eff=loss_target_eff,
                                            accept_threshold_eff_cand=loss_acceptable_eff_init,
                                            best_val_loss=best_val_loss,
                                            current_val_loss=current_val_loss,
                                            stageA_under_protest=bool(under_protest),
                                            best_train_loss=best_train_loss_initial,
                                            loss_scale=loss_scale,
                                            model_sep_output=model_sep_output,
                                            y_op=y_op,
                                            y_op_inv=y_op_inv,
                                            Nxvars=Nxvars,
                                            x_transform_map=x_transform_map,
                                            trig_spec=trig_spec,
                                            units_spec=units_spec,
                                            enforce_units=bool(enforce_units),
                                            units_reject_cb=_units_reject,
                                            oracle_trig_specs=None,
                                            allow_iterative_extension=atom_is_compound,
                                            skip_same_arity_if_already_sep=bool(early_already_sep),
                                            scaling_features=scale_specs,
                                        )

                                        if pd_accepted:
                                            _move_parent_ast = current_ast
                                            _move_parent_loss = best_val_loss
                                            model = pd_model
                                            current_ast = pd_ast
                                            best_val_loss = pd_loss
                                            current_val_loss = pd_loss
                                            _stageA_record_move(
                                                move_kind="early_power_difference_compound",
                                                parent_ast=_move_parent_ast,
                                                candidate_ast=current_ast,
                                                parent_loss=_move_parent_loss,
                                                candidate_loss=pd_loss,
                                                reason="early pure-difference/power-difference compound accepted",
                                                risk_tags={"compound_coordinate"},
                                                details=_stageA_compound_move_details(pd_model, pd_full),
                                            )
                                            _stageA_sync_shadow_registry(search_hp, current_ast, reason="early power-difference")
                                            separability_success = True
                                            print("[Early PureDiff] Accepted power-difference compound, continuing...")
                                            early_made_progress = True

                                            # --- Compound trig re-probe ---
                                            try:
                                                compound_targets_post = _extract_compound_targets_from_ast(current_ast)
                                                if compound_targets_post:
                                                    oracle_compound = probe_oracle_scaling(
                                                        model=model,
                                                        datagen=datagen_train_noshuffle,
                                                        Nxvars=Nxvars,
                                                        device=device,
                                                        compound_targets=compound_targets_post,
                                                    )
                                                    oracle_specs_merged = list(oracle_specs or []) + oracle_compound

                                                    trig_reprobe = probe_trig_scaling(
                                                        model=model,
                                                        datagen=datagen_train_noshuffle,
                                                        Nxvars=Nxvars,
                                                        device=device,
                                                        oracle_specs=oracle_specs_merged,
                                                        compound_targets=compound_targets_post,
                                                    )
                                                    trig_scale_specs = trig_reprobe

                                                    all_trig = [
                                                        ts for ts in trig_reprobe
                                                        if ts.rel_std < 0.10
                                                    ]
                                                    if all_trig:
                                                        best_trig = min(all_trig, key=lambda ts: ts.rel_std)

                                                        if best_trig.compound_name:
                                                            print(
                                                                f"[Compound Trig] Skipping {best_trig.trig_fn}("
                                                                f"{best_trig.omega:.4g}*z) on "
                                                                f"{best_trig.compound_name}: standalone "
                                                                f"trig(compound) is degenerate"
                                                            )
                                                            x_precond_applied.add("trig")
                                                        else:
                                                            print(
                                                                f"[Compound Trig] Per-axis x{best_trig.axis} "
                                                                f"(rel_std={best_trig.rel_std:.4f}) beats compound; "
                                                                f"deferring to x-preconditioning"
                                                            )
                                                    else:
                                                        print("[Compound Trig] No trig signal found")
                                            except Exception as e:
                                                print(f"[Compound Trig] Re-probe failed: {e}")

                                    else:
                                        print("[Early PureDiff] No valid power-difference proposals found.")
                except Exception as e:
                    print(f"[Early PureDiff] Detection failed: {e}")
            # --------------------------------------------------------------------

            # --- Detect low-degree polynomial structure in f(x) -------------
            # Skip if early compound was accepted
            if y_op is None:
                try:
                    print("Checking for low-degree polynomial fit in x (degree 2)...")
                    poly_spec = discover_poly_in_x(
                        model=model,
                        datagen=datagen_train_noshuffle,
                        Nxvars=Nxvars,
                        device=device,
                        degree=2,
                    )
                    if poly_spec is not None:
                        print(
                            f"Poly-in-x (deg {poly_spec.degree}) fit: "
                            f"rms_abs={poly_spec.rms_abs:.3e}, "
                            f"rms_rel={poly_spec.rms_rel:.3e}, "
                            f"n_terms={poly_spec.n_terms}, n={poly_spec.n_points}"
                        )
                    else:
                        print("No useful poly-in-x fit found.")
                except Exception as e:
                    print("Poly-in-x detection failed with error:", e)

                # --- Detect low-degree polynomial fit for f(x)^2 ------------
                try:
                    print("Checking for low-degree polynomial fit in f(x)^2 (degree 2)...")
                    poly_f2_spec = discover_poly_in_f2(
                        model=model,
                        datagen=datagen_train_noshuffle,
                        Nxvars=Nxvars,
                        device=device,
                        degree=2,
                    )
                    if poly_f2_spec is not None:
                        print(
                            f"Poly-in-f^2 (deg {poly_f2_spec.degree}) fit: "
                            f"rms_abs={poly_f2_spec.rms_abs:.3e}, "
                            f"rms_rel={poly_f2_spec.rms_rel:.3e}, "
                            f"n_terms={poly_f2_spec.n_terms}, n={poly_f2_spec.n_points}"
                        )
                    else:
                        print("No useful poly-in-f^2 fit found.")
                except Exception as e:
                    print("Poly-in-f^2 detection failed with error:", e)
            # --------------------------------------------------------------------

            # --- Rational poly/poly fit: f(x) ≈ P(x)/Q(x) ------------------
            if y_op is None:
                try:
                    print("Checking for rational poly/poly fit f(x) ≈ P(x)/Q(x)...")
                    rat_spec = discover_rational_poly(
                        model=model,
                        datagen=datagen_train_noshuffle,
                        Nxvars=Nxvars,
                        device=device,
                        # Include degree-3+ numerators: some common rational laws
                        # become degree-3 after clearing a squared variable from the
                        # denominator (e.g. (x1+x2)/(1+x1*x2/x0^2) -> x0^2*(x1+x2)/(x0^2+x1*x2)).
                        max_deg_num=4,
                        max_deg_den=4,
                        min_points=200,
                        rel_rms_threshold=1e-3,
                    )
                    if rat_spec is not None:
                        print(
                            f"Rational fit {rat_spec.name}: "
                            f"deg_num={rat_spec.deg_num}, deg_den={rat_spec.deg_den}, "
                            f"rms_abs={rat_spec.rms_abs:.3e}, rms_rel={rat_spec.rms_rel:.3e}, "
                            f"n_terms_num={rat_spec.n_terms_num}, n_terms_den={rat_spec.n_terms_den}, "
                            f"n={rat_spec.n_points}, sigma_min={rat_spec.sigma_min:.3e}, "
                            f"sigma_ratio={rat_spec.sigma_ratio:.3e}"
                        )
                    else:
                        print("No useful rational poly/poly fit found.")
                except Exception as e:
                    print("Rational poly/poly detection failed with error:", e)
            # --------------------------------------------------------------------

            # --- Discover trig-like behaviour along axes --------------------
            # FFT-based trig axis discovery (INFORMATIONAL — not used in logic).
            # Oracle-derived trig_axis_specs_all is the sole source of truth.
            trig_spec = None  # Initialize for later use (strongest axis)
            try:
                print("Checking for trig-like behaviour along coordinate axes...")
                fft_trig_specs = discover_trig_axes(
                    model=model,
                    datagen=datagen_train_noshuffle,
                    Nxvars=Nxvars,
                    device=device,
                    strength_threshold=5.0,
                )
                if fft_trig_specs:
                    for sp in fft_trig_specs:
                        print(
                            f"  [FFT hint] axis {sp.axis}: omega~{sp.omega:.3g}, "
                            f"strength={sp.strength:.3g}, span=[{sp.tmin:.3g},{sp.tmax:.3g}]"
                        )
                else:
                    print("  [FFT hint] No strong trig axis detected.")
            except Exception as e:
                print(f"Trig-axis FFT hint failed: {e}")

            # Derive strongest oracle trig axis for diagnostics (not used in control flow).
            trig_spec = trig_axis_specs_all[0] if trig_axis_specs_all else None
            if trig_spec is not None:
                print(
                    f"Oracle trig axis: j={trig_spec.axis}, "
                    f"omega≈{trig_spec.omega:.3g}, strength={trig_spec.strength:.3g}"
                )
                if len(trig_axis_specs_all) > 1:
                    print(f"  ({len(trig_axis_specs_all)} total oracle trig axes)")
            else:
                print("No oracle-derived trig axis.")

            # --- Discover K-based preferred directions ----------------------
            dirs = []  # Initialize for later use
            try:
                print("Discovering K-based preferred directions...")
                dirs = discover_model_directions(
                    model, topk=min(8, Nxvars), cos_thr=0.95, out_idx=0
                )
                print("Found {} preferred directions from K:".format(len(dirs)))
                for k, u in enumerate(dirs):
                    u_str = ", ".join("{:+.3f}".format(float(v)) for v in u)
                    print(f"  dir {k}: u=[{u_str}]")
            except Exception as e:
                print("K-based direction discovery failed with error:", e)
            # --------------------------------------------------------------------

            # --- Trig check along discovered K-dirs -------------------------
            try:
                # build a rough x0 as the mean over a few training batches
                xs = []
                for bi, batch in enumerate(
                    datagen_train_noshuffle()
                    if callable(datagen_train_noshuffle)
                    else datagen_train_noshuffle
                ):
                    if bi >= 2:
                        break
                    x = batch[0] if isinstance(batch, (list, tuple)) else batch
                    xs.append(x.view(x.size(0), -1).detach().cpu())
                if xs:
                    X_cache = torch.cat(xs, dim=0)
                    x_mean = X_cache.mean(dim=0)

                    for k, u in enumerate(dirs):
                        # If K lives in an internal (e.g. compound-leaf) input space,
                        # its directions may not match the raw x-dimension. In that
                        # case, skip this diagnostic instead of erroring out.
                        if int(getattr(u, "numel", lambda: len(u))()) != int(X_cache.shape[1]):
                            print(
                                f"  dir {k}: skipping trig check (dim mismatch: |u|={int(getattr(u, 'numel', lambda: len(u))())}, xdim={int(X_cache.shape[1])})"
                            )
                            continue
                        # choose a span using simple quantiles along u
                        t_proj = X_cache @ u
                        tmin = float(torch.quantile(t_proj, 0.05))
                        tmax = float(torch.quantile(t_proj, 0.95))
                        ts, f_line, d1_line, d2_line = sample_line_curvature(
                            provider=model,
                            x0=x_mean,
                            u=u,
                            tmin=tmin,
                            tmax=tmax,
                            n=512,
                            out_idx=0,
                        )
                        eta = 0.8 * float((ts[1:] - ts[:-1]).median())
                        P = poisson_profile(ts, d2_line, ts, eta)
                        dx = float(ts[1] - ts[0])
                        omega, strength = trig_from_profile(P, dx)
                        print(f"  dir {k}: omega≈{omega:.3g}, trig_strength≈{strength:.1f}")
            except Exception as e:
                print("Trig check along K-dirs failed with error:", e)
            # --------------------------------------------------------------------

            # --- Detect parity (even/odd) along coordinate axes ----------
            try:
                print("Checking for approximate even/odd parity along coordinate axes...")
                parity_specs = discover_parity_axes(
                    model=model,
                    datagen=datagen_train_noshuffle,
                    Nxvars=Nxvars,
                    device=device,
                )
                if parity_specs:
                    print(f"Detected parity candidates on {len(parity_specs)} axes:")
                    for ps in parity_specs:
                        print(
                            f"  axis={ps.axis:2d}  origin≈{ps.origin:8.3g}  kind={ps.kind:>4s}  "
                            f"rms_rel_even={ps.rms_rel_even:8.2e}  rms_rel_odd={ps.rms_rel_odd:8.2e}  "
                            f"n={ps.n_points}"
                        )
                else:
                    print("No strong parity patterns detected.")
            except Exception as e:
                print("Parity detection failed with error:", e)
            # --------------------------------------------------------------------

            # --- Detect radial / distance-like dependence over variable groups ---
            try:
                print("Checking for approximately radial dependence over small variable groups...")
                radial_specs = discover_radial_groups(
                    model=model,
                    datagen=datagen_train_noshuffle,
                    Nxvars=Nxvars,
                    device=device,
                    max_group_size=min(3, Nxvars),
                )
                if radial_specs:
                    print(f"Discovered {len(radial_specs)} radial candidates:")
                    for rs in radial_specs:
                        idx_str = ",".join(str(j) for j in rs.indices)
                        print(
                            f"  S=[{idx_str}]  mean|cos(∇f, x)|={rs.mean_abs_cos:6.3f}  "
                            f"med|cos|={rs.median_abs_cos:6.3f}  n={rs.n_points}"
                        )
                else:
                    print("No clear radial structure detected.")
            except Exception as e:
                print("Radial-structure detection failed with error:", e)
            # --------------------------------------------------------------------

            # --- Discover preferred origins (translation structure) ----------
            try:
                print("Checking for preferred origins / translation structure along axes...")
                origin_specs = discover_preferred_origins(
                    model=model,
                    datagen=datagen_train_noshuffle,
                    Nxvars=Nxvars,
                    device=device,
                )
                if origin_specs:
                    print(f"Discovered {len(origin_specs)} candidate preferred origins:")
                    for ts in origin_specs:
                        in_flag = "in-range" if ts.in_range else "out-of-range"
                        print(
                            f"  axis={ts.axis:2d}  x0≈{ts.origin:8.3g}  "
                            f"slope={ts.slope:9.3e}  r2={ts.r2:6.3f}  {in_flag}, n={ts.n_points}"
                        )
                else:
                    print("No strong translation structure detected.")
            except Exception as e:
                print("Preferred-origin detection failed with error:", e)
            # --------------------------------------------------------------------

            # --- Detect saturating / sigmoidal behaviour along axes ---------
            try:
                print("Checking for saturating / sigmoidal behaviour along coordinate axes...")
                sat_specs = discover_saturating_axes(
                    model=model,
                    datagen=datagen_train_noshuffle,
                    Nxvars=Nxvars,
                    device=device,
                )
                if sat_specs:
                    print(f"Discovered {len(sat_specs)} saturating-axis candidates:")
                    for ss in sat_specs:
                        dir_str = "increasing" if ss.direction > 0 else "decreasing"
                        print(
                            f"  axis={ss.axis:2d}  span=[{ss.tmin:8.3g},{ss.tmax:8.3g}]  "
                            f"mid/edge|f'|≈{ss.mid_edge_grad_ratio:6.2f}  "
                            f"mono_frac={ss.monotonic_fraction:5.2f}  {dir_str}, n={ss.n_points}"
                        )
                else:
                    print("No clear saturating behaviour detected along single axes.")
            except Exception as e:
                print("Saturating-axis detection failed with error:", e)
            # --------------------------------------------------------------------

            # --- Detect near-constant directions (parameter-like variables) ---
            try:
                print("Checking for near-constant directions (parameter-like variables)...")
                const_specs = discover_constant_directions(
                    model=model,
                    datagen=datagen_train_noshuffle,
                    Nxvars=Nxvars,
                    device=device,
                )
                if const_specs:
                    print(f"Discovered {len(const_specs)} parameter-like directions:")
                    for cs in const_specs:
                        print(
                            f"  axis={cs.axis:2d}  rms|∂f/∂x_j|={cs.rms_grad:9.3e}  "
                            f"rel_rms={cs.rel_rms_grad:7.3f}  n={cs.n_points}"
                        )
                else:
                    print("No strongly parameter-like directions detected.")
            except Exception as e:
                print("Constant-direction detection failed with error:", e)
            # --------------------------------------------------------------------

        # Trap case = single NN atom on variable 0
        if is_minimal_ast(current_ast):
            print("Model is in minimal form. Ending the Stage A run.")
            separability_success = True
            break

        # Initial separability-op check for identity transform
        # Skip if we already have multiple expressions (e.g., after x-preconditioning restart)
        if i == 0 and y_op is None and model._count_atoms(model.ast_root) == 1:
            precision = search_hp.precision_derivs_d2y * 100.0
            print("Initial separability check with precision {}".format(precision))
            if is_multi and models_multi is not None:
                flags_all: List[List[bool]] = []
                for di, m_i in enumerate(models_multi):
                    flags_i = check_separability_ops(
                        list(range(Nxvars)),
                        0,
                        m_i,
                        train_loaders_all[di],
                        precision_sum=precision,
                        precision_mult=precision,
                        device=device,
                        y_transform_names=y_transform_names,
                        very_verbose=verbose_sep,
                    )
                    flags_all.append(list(flags_i))
                if flags_all:
                    merged_flags = [all(f[k] for f in flags_all) for k in range(len(flags_all[0]))]
                else:
                    merged_flags = [False] * len(candidate_sep_ops)
                candidate_sep_ops_local = merged_flags
            else:
                candidate_sep_ops_local = check_separability_ops(
                    list(range(Nxvars)),
                    0,
                    model,
                    datagen_train_noshuffle,
                    precision_sum=precision,
                    precision_mult=precision,
                    device=device,
                    y_transform_names=y_transform_names,
                    very_verbose=verbose_sep,
                )
            for k, flag in enumerate(candidate_sep_ops_local):
                candidate_sep_ops[k] = bool(flag)

            if candidate_sep_ops[i_op]:
                print("Initial fit suggests separability candidates for y_op {}.".format(y_op_str))
            else:
                print(
                    "Initial fit suggests no separability candidates for y_op {}, continuing to compound detection.".format(
                        y_op_str
                    )
                )
                separable = False
                # Removed: break  - Allow compound detection to run as fallback

        if i == 0:
            print(f"\n{'='*70}")
            print("  [Stage A] SEPARABILITY SEARCH")
            print(f"{'='*70}")
        else:
            print(f"\n{'='*70}")
            print(f"  [Stage A] SEPARABILITY SEARCH — restart (pass {i+1})")
            print(f"{'='*70}")

        # ------------------------------------------------------------------
        #  Now handle AST atoms
        # ------------------------------------------------------------------
        changed = False

        # ==================================================================
        # PHASE A: Exhaust all compound opportunities before separability
        #
        # The algorithm: compounds take priority over separability. We loop
        # repeatedly looking for compounds until none are found, only then
        # do we try separability. If separability finds something, the outer
        # `while separable:` loop will bring us back here to look for more
        # compounds that might have been exposed by the new structure.
        # ==================================================================
        # ------------------------------------------------------------------
        # Stage A move-policy: disjoint separability vs long composite variables
        #
        # Stage A CPU can be dominated by compound detection / candidate training.
        # When there is *very clear* evidence that we can split an atom cleanly
        # (additive or multiplicative) with a disjoint cover of its variables,
        # it can be cheaper to do that first and let the simplified structure
        # expose better compound/separability opportunities later.
        #
        # This is a dynamic switch between the two extremes:
        #   - compound-first (greedy): exhaust high-confidence compounds before sep
        #   - separability-first (frugal): try only the best disjoint split first
        # ------------------------------------------------------------------
        stageA_move_policy = str(getattr(search_hp, "stageA_move_policy", "dynamic")).lower()

        # If we skip compound exhaustion, remember it; if separability makes no
        # progress we fall back to compound exhaustion later in this iteration.
        compound_exhaustion_deferred = False

        # When set, restrict separability candidates to a "safe" subset:
        #   - "singleton": singleton-vs-rest disjoint covers only
        #   - "disjoint" : any disjoint cover
        sep_filter_mode = None

        compound_exhaustion_loop_active = True
        enable_compound_global = bool(getattr(search_hp, "enable_compound_detection", False)) and (not is_multi)

        if stageA_move_policy in ("sep_first", "separability_first"):
            # Always attempt separability before compounds (no candidate filtering).
            if compound_exhaustion_loop_active and enable_compound_global:
                compound_exhaustion_deferred = True
            compound_exhaustion_loop_active = False
        elif stageA_move_policy.startswith("dynamic"):
            # Dynamic policies:
            #   - dynamic_singleton: only override compounds for a very clean singleton disjoint split
            #   - dynamic / dynamic_disjoint: override compounds for any very clean disjoint split
            if stageA_move_policy in ("dynamic_singleton", "dynamic_sep_singleton", "dynamic_single"):
                require_singleton = True
            elif stageA_move_policy in ("dynamic", "dynamic_disjoint", "dynamic_sep_disjoint"):
                require_singleton = False
            else:
                # Default for unknown "dynamic*" strings: be conservative.
                require_singleton = True

            if compound_exhaustion_loop_active and enable_compound_global:
                try:
                    best_m, best_atom, precision_quickscan, best_is_singleton = _stageA_best_disjoint_separability_metric(
                        model=model,
                        current_ast=current_ast,
                        nn_atoms=collect_nn_atoms(current_ast),
                        datagen_train_noshuffle=datagen_train_noshuffle,
                        device=device,
                        dtype=dtype,
                        search_hp=search_hp,
                        lm_hp=lm_hp,
                        y_op=y_op,
                        y_med=y_med,
                        y_mad=y_mad,
                        y_log_dynamic_range=y_log_dynamic_range,
                        require_singleton=require_singleton,
                    )
                    factor_single = float(getattr(search_hp, "stageA_singleton_sep_metric_factor", 0.30))
                    factor_non = float(getattr(search_hp, "stageA_non_singleton_sep_metric_factor", 0.30))
                    metric_factor = factor_single if bool(best_is_singleton) else factor_non

                    if (
                        best_m is not None
                        and precision_quickscan is not None
                        and float(best_m) <= metric_factor * float(precision_quickscan)
                    ):
                        compound_exhaustion_deferred = True
                        compound_exhaustion_loop_active = False
                        sep_filter_mode = "singleton" if require_singleton else "disjoint"
                        atom_desc = ""
                        if best_atom is not None:
                            try:
                                atom_desc = f" on NN{list(best_atom.var_idxs)}"
                            except Exception:
                                pass
                        print(
                            f"[Stage A] Move policy '{stageA_move_policy}': strong disjoint separability hint{atom_desc} "
                            f"(metric={float(best_m):.2e} <= {metric_factor:.2f}*precision={metric_factor*float(precision_quickscan):.2e}); "
                            f"deferring compound exhaustion."
                        )
                except Exception as e:
                    print(f"[Stage A] Move policy quickscan failed: {type(e).__name__}: {e}")

        stageA_monomial_attempted_this_pass = False
        _compound_prev_ast_repr = ast_to_human_readable(current_ast)
        while compound_exhaustion_loop_active:
            compound_found_this_pass = False
            nn_atoms = collect_nn_atoms(current_ast)
            tag_to_leaf = _build_tag_to_leaf_map(current_ast, model)

            # Extract compound detection parameters (same as below)
            enable_compound = bool(getattr(search_hp, "enable_compound_detection", False)) and (not is_multi)
            compound_max_vars = int(getattr(search_hp, "compound_max_vars", 4))
            compound_max_exponent = int(getattr(search_hp, "compound_max_exponent", 5))
            compound_threshold = float(getattr(search_hp, "compound_threshold", 0.05))
            compound_max_batches = int(getattr(search_hp, "compound_max_batches", 4))
            COMPOUND_CONF_THRESHOLD = float(getattr(search_hp, "compound_confidence_gate", 0.85))

            if not enable_compound:
                break  # Exit compound exhaustion loop if compound detection is disabled

            for atom in nn_atoms:
                if effective_arity(atom) <= 1:
                    stageA_monomial_attempted_this_pass = True
                    acc_m, new_model_m, new_ast_m, new_loss_m = _try_stageA_univariate_monomial_for_atom(
                        model=model,
                        current_ast=current_ast,
                        atom=atom,
                        tag_to_leaf=tag_to_leaf,
                        datagen_train_noshuffle=datagen_train_noshuffle,
                        datagen_val_noshuffle=datagen_val_noshuffle,
                        device=device,
                        dtype=dtype,
                        leaf_builder=leaf_builder,
                        dual_layer_used=dual_layer_used,
                        search_hp=search_hp,
                        lm_hp=lm_hp,
                        loss_target_eff=loss_target_eff,
                        accept_threshold_eff_cand=accept_threshold_eff_cand,
                        best_val_loss=best_val_loss,
                        current_val_loss=current_val_loss,
                        stageA_under_protest=bool(under_protest),
                        best_train_loss=best_train_loss_initial,
                        loss_scale=loss_scale,
                        units_spec=units_spec,
                        enforce_units=bool(enforce_units),
                        units_reject_cb=_units_reject,
                        y_op=y_op,
                        y_op_inv=y_op_inv,
                    )
                    if acc_m:
                        _ast_before = ast_to_human_readable(current_ast)
                        _ast_after = ast_to_human_readable(new_ast_m)
                        _move_parent_ast = current_ast
                        _move_parent_loss = current_val_loss
                        model = new_model_m
                        current_ast = new_ast_m
                        current_val_loss = new_loss_m
                        if _ast_after == _ast_before:
                            print(
                                "[Stage A Monomial] Accepted candidate kept AST unchanged; "
                                "ignoring no-op accept."
                            )
                            break
                        _stageA_record_move(
                            move_kind="stageA_monomial",
                            parent_ast=_move_parent_ast,
                            candidate_ast=current_ast,
                            parent_loss=_move_parent_loss,
                            candidate_loss=new_loss_m,
                            reason="Stage-A monomial candidate accepted",
                            risk_tags={"terminal_closure"} if not collect_nn_atoms(current_ast) else None,
                        )
                        _stageA_sync_shadow_registry(search_hp, current_ast, reason="Stage A monomial")
                        changed = True
                        compound_found_this_pass = True
                        break
                    continue

                if str(getattr(atom, "kind", "")).lower() != "nn":
                    continue

                if not (2 <= effective_arity(atom) <= compound_max_vars):
                    continue

                atom_already_compound = has_nontrivial_input(atom)
                leaf = tag_to_leaf.get(atom.tag)
                skip_same_arity_wrappers = False

                if atom_already_compound:
                    # Check if current compound already separates from extras.
                    # If so, skip same-arity coordinate churn and only let true
                    # arity-reducing extensions compete before Phase B peels the split.
                    inputs = get_input_exprs(atom)
                    z_expr_cur = inputs[0]
                    extra_var_idxs_cur = [int(inp.var_idxs[0]) for inp in inputs[1:] if is_trivial_input(inp)]
                    extra_input_asts_cur = [inp for inp in inputs[1:] if not is_trivial_input(inp)]
                    if (extra_var_idxs_cur or extra_input_asts_cur) and leaf is not None:
                        already_sep = _quick_separability_check(
                            model=model,
                            leaf=leaf,
                            z_expr=z_expr_cur,
                            extra_var_idxs=extra_var_idxs_cur,
                            extra_input_asts=extra_input_asts_cur,
                            datagen_train=datagen_train_noshuffle,
                            device=device,
                            dtype=dtype,
                        )
                        if already_sep:
                            if _should_skip_compound_extension_after_sep(
                                already_sep=True,
                                extra_var_idxs=extra_var_idxs_cur,
                                extra_input_asts=extra_input_asts_cur,
                            ):
                                print(
                                    f"[Compound] Skipping extension for NN{list(atom.var_idxs)}: "
                                    f"compound already separates from extras {extra_var_idxs_cur}"
                                )
                                continue  # Skip to next atom; Phase B will handle extras
                            print(
                                f"[Compound] Existing compound separates from extras {extra_var_idxs_cur}, "
                                "but checking only extensions that consume the separated coordinate."
                            )
                            skip_same_arity_wrappers = True
                    print(
                        f"[Variable Extension] Checking iterative extension for compound atom NN{list(atom.var_idxs)}..."
                    )
                else:
                    print(
                        f"[Variable Extension] Detecting compound structure for NN{list(atom.var_idxs)}..."
                    )

                try:
                    compound_proposals, compound_oracle_trig_specs = _detect_compound_variable_for_atom(
                        model=model,
                        atom=atom,
                        leaf=leaf,
                        datagen_train=datagen_train_noshuffle,
                        device=device,
                        max_exponent=compound_max_exponent,
                        precision=compound_threshold,
                        max_batches=compound_max_batches,
                        enable_linear=bool(getattr(search_hp, "compound_try_linear", True)),
                        max_linear_coeff=int(getattr(search_hp, "compound_linear_max_coeff", 2)),
                        enable_radial=bool(getattr(search_hp, "compound_try_radial", True)),
                        radial_max_group_size=int(getattr(search_hp, "compound_radial_max_group_size", 3)),
                        radial_cos_threshold=float(getattr(search_hp, "compound_radial_cos_threshold", 0.95)),
                        radial_try_sqrt=bool(getattr(search_hp, "compound_radial_try_sqrt", True)),
                        enable_shift=bool(getattr(search_hp, "compound_try_shift", True)),
                        shift_min_r2=float(getattr(search_hp, "compound_shift_min_r2", 0.85)),
                        shift_min_abs_slope=float(getattr(search_hp, "compound_shift_min_abs_slope", 1e-6)),
                        shift_require_in_range=bool(getattr(search_hp, "compound_shift_require_in_range", True)),
                        shift_max_axes_per_atom=int(getattr(search_hp, "compound_shift_max_axes_per_atom", 2)),
                        scaling_features=scale_specs,
                        invariance_features=invariance_feats,
                        trig_axis_specs=trig_axis_specs_all,
                        enable_mixed_compound=bool(getattr(search_hp, "compound_try_mixed", True)),
                        enable_retained_axis_wrappers=bool(
                            getattr(search_hp, "compound_try_retained_axis_wrappers", True)
                        ),
                        units_spec=units_spec,
                        enforce_units=bool(enforce_units),
                        shadow_registry=_stageA_shadow_registry(search_hp),
                        gs_cfg=getattr(search_hp, "gs_config", None),
                    )
                except Exception as e:
                    print(f"[Compound Exhaustion] Detection failed: {type(e).__name__}: {e}")
                    compound_proposals = []
                    compound_oracle_trig_specs = []

                phase_hint_props = _phase_hint_compound_proposals_for_atom(
                    getattr(search_hp, "phase_hints", []),
                    atom,
                    min_confidence=COMPOUND_CONF_THRESHOLD,
                )
                if phase_hint_props:
                    compound_proposals = list(compound_proposals or []) + list(phase_hint_props)
                    print(
                        f"[PhaseScan] Added {len(phase_hint_props)} phase carrier(s) "
                        "to Stage-A compound proposals."
                    )

                compound_proposals = _stageA_append_compound_replay_proposals(
                    compound_proposals or [],
                    search_hp=search_hp,
                    lm_hp=lm_hp,
                    current_ast=current_ast,
                    atom=atom,
                    Nxvars=Nxvars,
                    x_transform_map=x_transform_map,
                    units_spec=units_spec,
                )
                compound_proposals = _stageA_append_visible_buckingham_1d_prefactor_proposals(
                    compound_proposals,
                    current_ast=current_ast,
                    atom=atom,
                    units_spec=units_spec,
                    enforce_units=bool(enforce_units),
                    search_hp=search_hp,
                    x_transform_map=x_transform_map,
                )

                # Only try if the best proposal has confidence >= threshold.
                # Detection order can mix families; a weaker older proposal must
                # not suppress a later high-confidence metric-distance proposal.
                best_conf = _compound_best_proposal_confidence(compound_proposals)
                if compound_proposals and best_conf >= COMPOUND_CONF_THRESHOLD:
                    print(
                        f"[Compound Exhaustion] High-confidence proposal (conf={best_conf:.3f}), trying..."
                    )
                    acc_c, new_model, new_ast, new_loss, full_c, _enables_sep = _try_compound_candidates_for_atom(
                        proposals=compound_proposals,
                        model=model,
                        current_ast=current_ast,
                        atom=atom,
                        tag_to_leaf=tag_to_leaf,
                        datagen_train_noshuffle=datagen_train_noshuffle,
                        datagen_val_noshuffle=datagen_val_noshuffle,
                        device=device,
                        dtype=dtype,
                        leaf_builder=leaf_builder,
                        dual_layer_used=dual_layer_used,
                        search_hp=search_hp,
                        lm_hp=lm_hp,
                        loss_target_eff=loss_target_eff,
                        accept_threshold_eff_cand=accept_threshold_eff_cand,
                        best_val_loss=best_val_loss,
                        current_val_loss=current_val_loss,
                        stageA_under_protest=bool(under_protest),
                        best_train_loss=best_train_loss_initial,
                        loss_scale=loss_scale,
                        model_sep_output=model_sep_output,
                        y_op=y_op,
                        y_op_inv=y_op_inv,
                        Nxvars=Nxvars,
                        x_transform_map=x_transform_map,
                        trig_spec=trig_spec,
                        units_spec=units_spec,
                        enforce_units=bool(enforce_units),
                        units_reject_cb=_units_reject,
                        allow_iterative_extension=atom_already_compound,
                        skip_same_arity_if_already_sep=skip_same_arity_wrappers,
                        oracle_trig_specs=compound_oracle_trig_specs,
                        scaling_features=scale_specs,
                    )
                    if acc_c:
                        _ast_before = ast_to_human_readable(current_ast)
                        _ast_after = ast_to_human_readable(new_ast)
                        _move_parent_model = model
                        _move_parent_models_multi = (
                            list(models_multi) if models_multi is not None else None
                        )
                        _move_parent_val_losses = (
                            list(current_val_losses) if current_val_losses is not None else None
                        )
                        _move_parent_ast = current_ast
                        _move_parent_loss = current_val_loss
                        model = new_model
                        current_ast = new_ast
                        current_val_loss = new_loss
                        if _ast_after == _ast_before:
                            print(
                                "[Compound Exhaustion] Accepted candidate kept AST unchanged; "
                                "ignoring no-op accept."
                            )
                            break
                        _move_details = _stageA_compound_move_details(new_model, full_c)
                        _stageA_record_move(
                            move_kind="compound_exhaustion",
                            parent_ast=_move_parent_ast,
                            candidate_ast=current_ast,
                            parent_loss=_move_parent_loss,
                            candidate_loss=new_loss,
                            reason="compound exhaustion candidate accepted",
                            risk_tags={"compound_coordinate"},
                            details=_move_details,
                        )
                        _stageA_begin_pending_full_refit_transaction(
                            move_kind="compound_exhaustion",
                            parent_ast=_move_parent_ast,
                            parent_model=_move_parent_model,
                            parent_models_multi=_move_parent_models_multi,
                            parent_val_loss=_move_parent_loss,
                            parent_val_losses=_move_parent_val_losses,
                            candidate_ast=current_ast,
                            candidate_loss=new_loss,
                            details=_move_details,
                        )
                        _stageA_sync_shadow_registry(search_hp, current_ast, reason="compound exhaustion")
                        changed = True
                        compound_found_this_pass = True
                        (
                            term_ok,
                            term_model,
                            term_ast,
                            term_loss,
                            _term_label,
                        ) = _try_stageA_terminal_closure_probe(
                            model=model,
                            current_ast=current_ast,
                            current_val_loss=float(current_val_loss),
                            datagen_train_noshuffle=datagen_train_noshuffle,
                            datagen_val_noshuffle=datagen_val_noshuffle,
                            device=device,
                            dtype=dtype,
                            search_hp=search_hp,
                            lm_hp=lm_hp,
                            loss_target_eff=loss_target_eff,
                            loss_scale=loss_scale,
                            model_sep_output=model_sep_output,
                            y_op=y_op,
                            y_op_inv=y_op_inv,
                            Nxvars=Nxvars,
                            dual_layer_used=dual_layer_used,
                            x_transform_map=x_transform_map,
                            units_spec=units_spec,
                            enforce_units=bool(enforce_units),
                        )
                        if term_ok:
                            _move_parent_ast = current_ast
                            _move_parent_loss = current_val_loss
                            model = term_model
                            current_ast = term_ast
                            current_val_loss = term_loss
                            _stageA_record_move(
                                move_kind="terminal_closure_after_compound_exhaustion",
                                parent_ast=_move_parent_ast,
                                candidate_ast=current_ast,
                                parent_loss=_move_parent_loss,
                                candidate_loss=term_loss,
                                reason="terminal closure accepted after compound exhaustion",
                                risk_tags={"terminal_closure"},
                                details={"label": str(_term_label)},
                            )
                            full_compound_solved = True
                        if full_c:
                            print(
                                "[Compound Exhaustion] Full-variable compound compressed; outer map still unresolved"
                            )
                            full_compound_solved = True
                        break  # Restart compound loop with new AST

            if not compound_found_this_pass:
                # No compound found this pass, exit compound exhaustion loop
                break

            # Identical-state guard: if the AST is unchanged after "accepting"
            # a compound, we're stuck (e.g. passthrough re-accepted).  Break.
            _compound_cur_repr = ast_to_human_readable(current_ast)
            if _compound_cur_repr == _compound_prev_ast_repr:
                print("[Compound Exhaustion] AST unchanged after accept; breaking to avoid infinite loop.")
                break
            _compound_prev_ast_repr = _compound_cur_repr

        # NOTE: We don't continue here! After compounds are exhausted, we proceed
        # to Phase B (separability). If separability finds something, the main
        # `while separable:` loop will bring us back to try more compounds.
        #
        # Before ordinary separability, try a block-level additive shared-response
        # transaction. This handles cases where direct additive NN siblings share
        # one visible unit-certified scalar response, but individual sibling
        # gauges fail the local P*NN[pi] iso-z null test.
        if (
            (not changed)
            and enable_compound_global
            and (not is_multi)
            and bool(getattr(search_hp, "additive_shared_response_enable", True))
        ):
            tag_to_leaf_asr = _build_tag_to_leaf_map(current_ast, model)
            (
                acc_asr,
                new_model_asr,
                new_ast_asr,
                new_loss_asr,
                asr_details,
            ) = _try_stageA_additive_shared_response_block(
                model=model,
                current_ast=current_ast,
                tag_to_leaf=tag_to_leaf_asr,
                datagen_train_noshuffle=datagen_train_noshuffle,
                datagen_val_noshuffle=datagen_val_noshuffle,
                device=device,
                dtype=dtype,
                leaf_builder=leaf_builder,
                dual_layer_used=dual_layer_used,
                search_hp=search_hp,
                lm_hp=lm_hp,
                loss_target_eff=loss_target_eff,
                accept_threshold_eff_cand=accept_threshold_eff_cand,
                best_val_loss=best_val_loss,
                current_val_loss=current_val_loss,
                stageA_under_protest=bool(under_protest),
                best_train_loss=best_train_loss_initial,
                loss_scale=loss_scale,
                y_op=y_op,
                y_op_inv=y_op_inv,
                units_spec=units_spec,
                enforce_units=bool(enforce_units),
                x_transform_map=x_transform_map,
                data_hp=data_hp,
            )
            if acc_asr:
                _ast_before = ast_to_human_readable(current_ast)
                _ast_after = ast_to_human_readable(new_ast_asr)
                _move_parent_model = model
                _move_parent_models_multi = (
                    list(models_multi) if models_multi is not None else None
                )
                _move_parent_val_losses = (
                    list(current_val_losses) if current_val_losses is not None else None
                )
                _move_parent_ast = current_ast
                _move_parent_loss = current_val_loss
                model = new_model_asr
                current_ast = new_ast_asr
                current_val_loss = new_loss_asr
                if _ast_after == _ast_before:
                    print(
                        "[Stage A AdditiveSharedResponse] Accepted candidate kept AST unchanged; "
                        "ignoring no-op accept."
                    )
                else:
                    _stageA_record_move(
                        move_kind="additive_shared_response",
                        parent_ast=_move_parent_ast,
                        candidate_ast=current_ast,
                        parent_loss=_move_parent_loss,
                        candidate_loss=new_loss_asr,
                        reason="unit-certified additive shared-response candidate accepted",
                        risk_tags={"compound_coordinate", "additive_shared_response"},
                        details=asr_details,
                    )
                    _stageA_begin_pending_full_refit_transaction(
                        move_kind="additive_shared_response",
                        parent_ast=_move_parent_ast,
                        parent_model=_move_parent_model,
                        parent_models_multi=_move_parent_models_multi,
                        parent_val_loss=_move_parent_loss,
                        parent_val_losses=_move_parent_val_losses,
                        candidate_ast=current_ast,
                        candidate_loss=new_loss_asr,
                        details=asr_details,
                    )
                    _stageA_sync_shadow_registry(search_hp, current_ast, reason="additive shared response")
                    changed = True
                    separability_success = True
                    _clear_plain_random_branch_after_structural_accept("additive shared response")
                    _refresh_fit_link_original_y_certificate("additive shared response", quiet=True)
                    i += 1
                    if _stageA_should_restart("additive shared response"):
                        separable = True
                        continue
                    separable = False
                    break

        # Monomial peels are cheaper and less gauge-ambiguous than full compound
        # exhaustion, so they should still run when the move policy deliberately
        # defers compounds in favour of a strong separability hint.
        stageA_monomial_accepted_this_pass = False
        if not stageA_monomial_attempted_this_pass:
            nn_atoms_m = collect_nn_atoms(current_ast)
            tag_to_leaf_m = _build_tag_to_leaf_map(current_ast, model)
            for atom_m in nn_atoms_m:
                if effective_arity(atom_m) > 1:
                    continue
                acc_m, new_model_m, new_ast_m, new_loss_m = _try_stageA_univariate_monomial_for_atom(
                    model=model,
                    current_ast=current_ast,
                    atom=atom_m,
                    tag_to_leaf=tag_to_leaf_m,
                    datagen_train_noshuffle=datagen_train_noshuffle,
                    datagen_val_noshuffle=datagen_val_noshuffle,
                    device=device,
                    dtype=dtype,
                    leaf_builder=leaf_builder,
                    dual_layer_used=dual_layer_used,
                    search_hp=search_hp,
                    lm_hp=lm_hp,
                    loss_target_eff=loss_target_eff,
                    accept_threshold_eff_cand=accept_threshold_eff_cand,
                    best_val_loss=best_val_loss,
                    current_val_loss=current_val_loss,
                    stageA_under_protest=bool(under_protest),
                    best_train_loss=best_train_loss_initial,
                    loss_scale=loss_scale,
                    units_spec=units_spec,
                    enforce_units=bool(enforce_units),
                    units_reject_cb=_units_reject,
                    y_op=y_op,
                    y_op_inv=y_op_inv,
                )
                if acc_m:
                    _ast_before = ast_to_human_readable(current_ast)
                    _ast_after = ast_to_human_readable(new_ast_m)
                    _move_parent_ast = current_ast
                    _move_parent_loss = current_val_loss
                    model = new_model_m
                    current_ast = new_ast_m
                    current_val_loss = new_loss_m
                    if _ast_after == _ast_before:
                        print(
                            "[Stage A Monomial] Accepted candidate kept AST unchanged; "
                            "ignoring no-op accept."
                        )
                        break
                    _stageA_record_move(
                        move_kind="stageA_monomial_restart",
                        parent_ast=_move_parent_ast,
                        candidate_ast=current_ast,
                        parent_loss=_move_parent_loss,
                        candidate_loss=new_loss_m,
                        reason="Stage-A monomial candidate accepted before restart",
                        risk_tags={"terminal_closure"} if not collect_nn_atoms(current_ast) else None,
                    )
                    _stageA_sync_shadow_registry(search_hp, current_ast, reason="Stage A monomial")
                    changed = True
                    separable = True
                    stageA_monomial_accepted_this_pass = True
                    print("[Stage A Monomial] Restarting Stage A after accepted monomial peel.")
                    break
            if stageA_monomial_accepted_this_pass:
                continue

        # ==================================================================
        # PHASE B: Now try separability (compounds are exhausted)
        # ==================================================================
        nn_atoms = collect_nn_atoms(current_ast)
        done_atoms = set()  # Track which atoms are done using object identity
        # Build tag->leaf map once per iteration (robust to FreeConst atoms)
        tag_to_leaf = _build_tag_to_leaf_map(current_ast, model)

        for atom in nn_atoms:
            # Skip if already "done"
            if id(atom) in done_atoms:
                continue

            # Single variable -> irreducible for separability purposes
            if effective_arity(atom) <= 1:
                done_atoms.add(id(atom))
                continue

            # Compute precision and ask for splits. We re-use this same
            # scale as the tolerance in the per-leaf non-dependency probe.
            precision = precision_for_transform(
                y_op=y_op,
                y_med=y_med,
                y_mad=y_mad,
                base_precision=search_hp.precision_derivs_d2y,
            )

            # Adjust precision for asinh fit-link: errors in asinh-space map to larger
            # errors in y-space, especially for second derivatives across high dynamic range.
            # The asinh inverse derivative is sqrt(1 + y^2/s^2), so at the scale point
            # the factor is ~sqrt(2), and for second derivatives across wide range the
            # factor scales with the log dynamic range.
            if getattr(lm_hp, "fit_y_link", None) == "asinh":
                asinh_scale = getattr(lm_hp, "fit_y_link_scale", 1.0)
                # Base factor from asinh inverse derivative at scale point
                base_factor = math.sqrt(1.0 + (asinh_scale / (y_mad + 1e-30)) ** 2)
                # Scale by dynamic range for second derivative noise across wide range
                dynamic_range_factor = y_log_dynamic_range if y_log_dynamic_range is not None else 2.0
                precision_factor = base_factor * dynamic_range_factor
                precision *= precision_factor
                print(f"{YELLOW}[asinh] Adjusting separability precision by {precision_factor:.2f}x to {precision:.4f}{RESET}")

            # ----------------------------------------------------------
            # 1) Leaf-level non-dependency prune:
            #
            #    If the leaf's own gradient wrt some input axes is
            #    numerically tiny, propose NN[x0,x1,x2] → NN[x0,x2],
            #    train a candidate, and accept it if the fit does not
            #    degrade beyond a small factor.
            # ----------------------------------------------------------
            axes_to_drop = _detect_leaf_nondep_axes_for_atom(
                model=model,
                atom=atom,
                leaf=tag_to_leaf.get(atom.tag),
                datagen_train=datagen_train_noshuffle,
                device=device,
                base_precision=precision,
            )
            if (not is_multi) and axes_to_drop:
                cand_ast_nd = _build_leaf_prune_candidate_ast(current_ast, atom, axes_to_drop)
                if cand_ast_nd is not None:
                    # Size estimate and num_segments choice (as in separability).
                    trial_model_size_target = model_hp.model_size_target
                    _, model_size_nd, _ = build_composite_ast(
                        cand_ast_nd,
                        model_hp.num_segments_min,  # dummy: size only
                        dual_layer=dual_layer_used,
                        leaf_builder=leaf_builder,
                        device=device,
                        dtype=dtype,
                    )
                    num_segments_nd = max(
                        min(trial_model_size_target // model_size_nd, model_hp.num_segments_max),
                        model_hp.num_segments_min,
                    )
                    num_segments_nd = cap_num_segments(
                        num_segments_nd, model_hp.num_segments_min, model_size_nd
                    )
                    print(
                        "Setting number of segments (leaf non-dep prune): {}".format(
                            num_segments_nd
                        )
                    )

                    # Reuse existing leaves keyed by tag, but ONLY if the candidate atom
                    # keeps the same var_idxs. (Leaf-prune changes var_idxs by design.)
                    # Clone leaves so rejected candidates don't mutate the current model.
                    tag_to_leaf = _build_tag_to_leaf_map(current_ast, model)
                    old_nn_atoms = collect_nn_atoms(current_ast)
                    cand_nn_atoms = collect_nn_atoms(cand_ast_nd)
                    cand_by_tag = {a.tag: a for a in cand_nn_atoms if getattr(a, "tag", None) is not None}
                    reuse_map_nd_raw = {}
                    for old_atom in old_nn_atoms:
                        tag = getattr(old_atom, "tag", None)
                        if tag is None or tag not in tag_to_leaf:
                            continue
                        cand_atom = cand_by_tag.get(tag)
                        if cand_atom is None:
                            continue
                        if tuple(getattr(cand_atom, "var_idxs", ())) != tuple(getattr(old_atom, "var_idxs", ())):
                            # Tag matches but axes differ -> do NOT reuse this leaf module.
                            continue
                        reuse_map_nd_raw[tag] = tag_to_leaf[tag]
                    reuse_leaves_nd = _clone_reuse_leaves(reuse_map_nd_raw, device, dtype)

                    # Capture the updated AST (cand_ast_nd_updated) to use when accepting
                    temp_model_nd, _, cand_ast_nd_updated = build_composite_ast(
                        cand_ast_nd,
                        num_segments_nd,
                        dual_layer=dual_layer_used,
                        leaf_builder=leaf_builder,
                        device=device,
                        dtype=dtype,
                        reuse_leaves=reuse_leaves_nd,
                        freeze_non_nn=freeze_non_nn,
                    )
                    temp_model_nd = _apply_fit_link_to_model(temp_model_nd, lm_hp)
                    if model_hp.nparam_max is not None:
                        temp_params_nd = temp_model_nd.num_parameters()
                        if temp_params_nd > model_hp.nparam_max:
                            print(
                                "Warning: leaf non-dep candidate model has {} parameters "
                                "exceeding nparam_max {}.".format(
                                    temp_params_nd, model_hp.nparam_max
                                )
                            )

                    # Acceptance threshold: shared loss-budget policy.
                    #
                    # Leaf non-dependency pruning is a simplification move (reduced effective arity
                    # and typically fewer parameters). We allow a limited temporary loss regression
                    # proportional to the simplification, and we also keep the historical
                    # partial_max_worsening_factor as a hard ceiling.
                    worsening_floor_nd = getattr(search_hp, "worsening_floor", 1.0e-6) * loss_scale
                    acceptance_noise_floor_raw = _resolve_acceptance_noise_floor_raw(lm_hp, loss_scale)
                    hard_ceiling_nd = _loss_excess_above_floor(global_ceil, acceptance_noise_floor_raw)
                    accept_threshold_nd = _compute_accept_threshold(
                        base_loss=best_val_loss,
                        best_loss=best_val_loss,
                        base_ast=current_ast,
                        cand_ast=cand_ast_nd_updated,
                        base_params=int(model.num_parameters()),
                        cand_params=int(temp_model_nd.num_parameters()),
                        loss_floor=float(loss_target_eff),
                        loss_cap=float(accept_threshold_eff_cand),
                        count_weight=float(getattr(lm_hp, "select_count_weight", 1.0)),
                        struct_gamma=float(getattr(lm_hp, "select_struct_gamma", 0.05)),
                        param_gamma=float(getattr(lm_hp, "select_param_gamma", 0.30)),
                        base_bonus_decades=float(getattr(lm_hp, "select_base_bonus_decades", 0.0)),
                        sep_bonus_decades=float(getattr(lm_hp, "select_sep_bonus_decades", 0.05)),
                        partial_sep_bonus_decades=float(getattr(lm_hp, "select_partial_sep_bonus_decades", 0.02)),
                        is_separability=False,
                        max_worsening_factor=float(partial_max_worsening_factor),
                        worsening_floor=float(worsening_floor_nd),
                        hard_ceiling=hard_ceiling_nd,
                        noise_floor=float(acceptance_noise_floor_raw),
                    )
                    accept_threshold_nd, structural_target_nd = _accept_threshold_with_structural_target(
                        base_ast=current_ast,
                        cand_ast=cand_ast_nd_updated,
                        accept_threshold=accept_threshold_nd,
                        loss_target_eff=loss_target_eff,
                    )
                    accept_threshold_nd, under_protest_cap_nd = _stageA_under_protest_threshold_cap(
                        accept_threshold=accept_threshold_nd,
                        current_val_loss=current_val_loss if current_val_loss is not None else best_val_loss,
                        loss_floor=loss_target_eff,
                        noise_floor=acceptance_noise_floor_raw,
                        under_protest=bool(under_protest),
                        label="leaf non-dependency prune",
                    )

                    proposed_str_nd = ast_to_human_readable(cand_ast_nd_updated)
                    print(
                        "***Training candidate leaf non-dependency-pruned model***, "
                        "acceptance threshold (raw) {:.4e} (base {:.4e})\n"
                        "    Proposed: {}".format(
                            accept_threshold_nd, search_hp.loss_acceptable,
                            proposed_str_nd
                        )
                    )
                    if structural_target_nd:
                        print(
                            "[Stage A] Structural arity reduction target enabled: "
                            f"arity signature {_nn_split_signature(current_ast)}"
                            f" → {_nn_split_signature(cand_ast_nd_updated)}, "
                            f"target-quality threshold {accept_threshold_nd:.4e}"
                        )
                    if under_protest_cap_nd:
                        print("[Stage A] Under-protest branch: requiring non-regressing validation loss.")
                    try:
                        accepted_nd, best_val_loss_nd, best_train_loss_nd, best_param_vec_nd, lm_opt_nd = fit_stageA_candidate_with_tournament(
                            temp_model_nd,
                            datagen_train_noshuffle,
                            datagen_val_noshuffle,
                            epochs=lm_hp.epochs,
                            LM_strategy=lm_hp.strategy,
                            nval_patience=lm_hp.nval_patience,
                            loss_target=loss_target_eff,
                            accept_threshold=accept_threshold_nd,
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
                            lm_hp=lm_hp,
                        )
                    except RuntimeError as e:
                        print(f"{YELLOW}Leaf non-dep prune candidate crashed; rejecting.{RESET} ({type(e).__name__}: {e})")
                        accepted_nd, best_val_loss_nd, best_train_loss_nd, best_param_vec_nd, lm_opt_nd = False, float('inf'), float('inf'), None, None

                    projection_override_nd = False
                    if not accepted_nd:
                        base_val_for_projection = (
                            current_val_loss if current_val_loss is not None else best_val_loss
                        )
                        max_train_degradation_nd = float(
                            getattr(search_hp, "max_train_degradation", 100.0)
                        )
                        projection_ok_nd, projection_reason_nd = (
                            _stageA_leaf_projection_nonregression_override(
                                base_ast=current_ast,
                                cand_ast=cand_ast_nd_updated,
                                base_val_loss=base_val_for_projection,
                                cand_val_loss=best_val_loss_nd,
                                loss_floor=loss_target_eff,
                                noise_floor=acceptance_noise_floor_raw,
                                base_train_loss=best_train_loss_initial,
                                cand_train_loss=best_train_loss_nd,
                                max_train_degradation=max_train_degradation_nd,
                                axes_to_drop=axes_to_drop,
                            )
                        )
                        if projection_ok_nd and best_param_vec_nd is not None and lm_opt_nd is not None:
                            accepted_nd = True
                            projection_override_nd = True
                            print(
                                f"{GREEN}[Stage A] Accepting leaf non-dependency prune "
                                f"by projection non-regression:{RESET} {projection_reason_nd}"
                            )
                        elif projection_ok_nd:
                            print(
                                f"{YELLOW}[Stage A] Leaf projection non-regression passed "
                                "but best parameters were unavailable; rejecting candidate."
                                f"{RESET}"
                            )

                    if accepted_nd:
                        base_val_for_leaf_gate = (
                            current_val_loss if current_val_loss is not None else best_val_loss
                        )
                        try:
                            leaf_gate_n_eff = getattr(lm_hp, "acceptance_noise_n_eff", None)
                            if leaf_gate_n_eff is None:
                                leaf_gate_n_eff = _loader_n_eff(datagen_val_noshuffle)
                        except Exception:
                            leaf_gate_n_eff = None
                        leaf_gate_ok, leaf_gate_reason = _stageA_leaf_prune_acceptance_gate(
                            base_ast=current_ast,
                            cand_ast=cand_ast_nd_updated,
                            axes_to_drop=axes_to_drop,
                            base_val_loss=base_val_for_leaf_gate,
                            cand_val_loss=best_val_loss_nd,
                            loss_floor=loss_target_eff,
                            noise_floor=acceptance_noise_floor_raw,
                            n_eff=leaf_gate_n_eff,
                        )
                        if leaf_gate_ok:
                            print(f"[Leaf prune gate] {leaf_gate_reason}")
                        else:
                            print(f"{YELLOW}[Leaf prune gate] {leaf_gate_reason}{RESET}")
                            accepted_nd = False

                    coe_prune_gate_summary = None
                    coe_prune_gate_reason = None
                    if accepted_nd:
                        coe_gate_ok, coe_prune_gate_reason, coe_prune_gate_summary = (
                            _stageA_destructive_prune_committee_gate(
                                base_ast=current_ast,
                                cand_ast=cand_ast_nd_updated,
                                axes_to_drop=axes_to_drop,
                                filepath=filepath,
                                np_dtype=np_dtype,
                                dtype=dtype,
                                device=device,
                                data_hp=data_hp,
                                model_hp=model_hp,
                                lm_hp=lm_hp,
                                leaf_builder=leaf_builder,
                                y_op=y_op,
                                y_op_inv=y_op_inv,
                                dual_layer_used=dual_layer_used,
                                num_segments=num_segments_nd,
                            )
                        )
                        if coe_prune_gate_summary and coe_prune_gate_summary.get("enabled"):
                            if coe_gate_ok:
                                print(f"[CoE StageA prune gate] {coe_prune_gate_reason}")
                            else:
                                print(f"{YELLOW}[CoE StageA prune gate] {coe_prune_gate_reason}{RESET}")
                                accepted_nd = False

                    if accepted_nd:
                        lm_opt_nd._update_param_groups(best_param_vec_nd)
                        print(
                            f"{GREEN}Accepted{RESET} leaf non-dependency prune on "
                            f"NN{list(atom.var_idxs)} → NN{[v for v in atom.var_idxs if v not in axes_to_drop]}, "
                            f"val-loss {best_val_loss_nd:.4e}"
                            + (" [projection-nonregression]" if projection_override_nd else "")
                        )
                        model = temp_model_nd
                        # Use the updated AST (with correct per-atom kwargs) from build_composite_ast
                        _move_parent_ast = current_ast
                        _move_parent_loss = current_val_loss
                        current_ast = cand_ast_nd_updated
                        current_val_loss = best_val_loss_nd
                        _stageA_record_move(
                            move_kind="leaf_non_dependency_prune",
                            parent_ast=_move_parent_ast,
                            candidate_ast=current_ast,
                            parent_loss=_move_parent_loss,
                            candidate_loss=best_val_loss_nd,
                            reason=(
                                str(leaf_gate_reason)
                                + (
                                    f"; {coe_prune_gate_reason}"
                                    if coe_prune_gate_reason
                                    else ""
                                )
                            ),
                            risk_tags={"destructive_prune", "axis_deletion"},
                            details={
                                "atom_var_idxs": [int(v) for v in getattr(atom, "var_idxs", ())],
                                "axes_to_drop": [int(v) for v in axes_to_drop],
                                "projection_nonregression": bool(projection_override_nd),
                                "coe_stageA_prune_gate": coe_prune_gate_summary,
                            },
                        )

                        # Show current expression structure
                        try:
                            expr_str = _compact_expression_repr(
                                current_ast, max_length=240, y_op_inv=y_op_inv
                            )
                            print(f"[Stage A]   Current: {expr_str}")
                        except Exception as e:
                            print(f"[Stage A]   Expression display error: {e}")
                            # Fallback: try without y_op_inv
                            try:
                                expr_str = _compact_expression_repr(
                                    current_ast, max_length=240, y_op_inv=None
                                )
                                print(f"[Stage A]   Current: {expr_str}")
                            except Exception:
                                pass

                        changed = True
                        any_sep_split = True
                        # Restart outer loop from the simplified AST.
                        break
                    else:
                        print(
                            f"{RED}Rejected{RESET} leaf non-dependency prune on NN{list(atom.var_idxs)}, val-loss {best_val_loss_nd:.4e}"
                        )

            # ----------------------------------------------------------
            # 2) Residual re-fit: remove additive gauge contamination.
            #
            # When an atom is a direct additive term and all sibling
            # additive terms are fully decomposed (every NN leaf has
            # effective_arity <= 1), the jointly-trained atom may carry
            # gauge contamination from the shared variables.  Re-training
            # a standalone NN on the explicit residual y - sibling_output
            # removes this gauge freedom and gives the separability check
            # a clean signal.
            # ----------------------------------------------------------
            residual_refit_model = None
            residual_refit_ast = None
            refit_siblings = _find_residual_refit_context(current_ast, atom)
            if (not is_multi) and refit_siblings is not None and int(effective_arity(atom)) > 1:
                print(
                    f"[Residual Re-fit] NN{list(atom.var_idxs)} has fully-decomposed "
                    f"additive siblings; re-training on explicit residual."
                )
                try:
                    # Collect training data and compute sibling outputs.
                    x_chunks, y_chunks = [], []
                    dl = datagen_train_noshuffle() if callable(datagen_train_noshuffle) else datagen_train_noshuffle
                    with torch.no_grad():
                        for batch in dl:
                            xb = batch[0].to(device=device, dtype=dtype)
                            yb = batch[1].to(device=device, dtype=dtype)
                            x_chunks.append(xb)
                            y_chunks.append(yb)
                    x_cat = torch.cat(x_chunks, dim=0)
                    y_cat = torch.cat(y_chunks, dim=0)
                    if y_cat.dim() == 2:
                        y_cat = y_cat[:, 0]

                    with torch.no_grad():
                        sib_sum = torch.zeros(x_cat.shape[0], device=device, dtype=dtype)
                        for sib in refit_siblings:
                            sib_sum = sib_sum + _eval_ast_subtree_on_data(sib, tag_to_leaf, x_cat)
                    y_residual = y_cat - sib_sum

                    resid_mag = float(torch.median(torch.abs(y_residual)).item())
                    sib_mag = float(torch.median(torch.abs(sib_sum)).item())
                    print(
                        f"[Residual Re-fit] |residual| median={resid_mag:.4e}, "
                        f"|sibling| median={sib_mag:.4e}, "
                        f"ratio={resid_mag / (sib_mag + 1e-30):.4e}"
                    )

                    # Build a standalone single-atom model.
                    parent_num_segments = (atom.kwargs or {}).get(
                        "num_segments", search_hp.num_segments_map[dual_layer_used]
                    )
                    parent_dual_layer = (atom.kwargs or {}).get("dual_layer", dual_layer_used)
                    standalone_ast = AtomNode(
                        kind="nn",
                        var_idxs=atom.var_idxs,
                        kwargs=dict(atom.kwargs or {}),
                        tag=getattr(atom, "tag", None),
                        inputs=atom.inputs,
                    )
                    standalone_model, _, _ = build_composite_ast(
                        standalone_ast,
                        parent_num_segments,
                        dual_layer=parent_dual_layer,
                        leaf_builder=leaf_builder,
                        device=device,
                        dtype=dtype,
                    )
                    standalone_model = _apply_fit_link_to_model(standalone_model, lm_hp)

                    # Compute residual on validation data for separate val loader.
                    x_val_chunks, y_val_chunks = [], []
                    dl_val = datagen_val_noshuffle() if callable(datagen_val_noshuffle) else datagen_val_noshuffle
                    with torch.no_grad():
                        for batch in dl_val:
                            x_val_chunks.append(batch[0].to(device=device, dtype=dtype))
                            y_val_chunks.append(batch[1].to(device=device, dtype=dtype))
                    x_val_cat = torch.cat(x_val_chunks, dim=0)
                    y_val_cat = torch.cat(y_val_chunks, dim=0)
                    if y_val_cat.dim() == 2:
                        y_val_cat = y_val_cat[:, 0]
                    with torch.no_grad():
                        sib_sum_val = torch.zeros(x_val_cat.shape[0], device=device, dtype=dtype)
                        for sib in refit_siblings:
                            sib_sum_val = sib_sum_val + _eval_ast_subtree_on_data(sib, tag_to_leaf, x_val_cat)
                    y_resid_val = y_val_cat - sib_sum_val

                    # Create separate train / val residual data loaders.
                    resid_ds_train = TensorDataset(x_cat, y_residual.unsqueeze(-1))
                    resid_ds_val = TensorDataset(x_val_cat, y_resid_val.unsqueeze(-1))
                    resid_dl_train = DataLoader(resid_ds_train, batch_size=min(x_cat.shape[0], 4096), shuffle=False)
                    resid_dl_val = DataLoader(resid_ds_val, batch_size=min(x_val_cat.shape[0], 4096), shuffle=False)

                    # Train on the residual (generous acceptance — we always use the result).
                    _, refit_val_loss, _, refit_param_vec, refit_lm_opt = train_candidate_model(
                        standalone_model,
                        resid_dl_train,
                        resid_dl_val,
                        epochs=lm_hp.epochs,
                        LM_strategy=lm_hp.strategy,
                        nval_patience=lm_hp.nval_patience,
                        loss_target=loss_target_eff,
                        accept_threshold=float("inf"),
                        epochs_min=max(lm_hp.epochs_min, 50),
                        chisq_tol=lm_hp.chisq_tol,
                        device=device,
                        epochs_awful_check=None,
                        awful_threshold=None,
                        log_file=lm_hp.log_file,
                        log_to_console=lm_hp.log_to_console,
                        log_level=lm_hp.log_level,
                        lm_verbose=lm_hp.LM_verbose,
                        lm_hp=lm_hp,
                    )
                    if refit_param_vec is not None:
                        refit_lm_opt._update_param_groups(refit_param_vec)
                    residual_refit_model = standalone_model
                    residual_refit_ast = standalone_ast
                    print(
                        f"[Residual Re-fit] Standalone model trained, "
                        f"val-loss={refit_val_loss:.4e}"
                    )
                except Exception as e:
                    print(f"[Residual Re-fit] Failed: {type(e).__name__}: {e}")
                    residual_refit_model = None
                    residual_refit_ast = None

            # ----------------------------------------------------------
            # 3) Classical add/mult separability checks (second order).
            #
            # Unified path: check separability in the atom's input space
            # using _check_separability_in_input_space for both trivial
            # and compound atoms.
            #
            # When a residual re-fit model is available (multi-atom case),
            # use its leaf for a cleaner signal.
            # ----------------------------------------------------------
            atom_has_compound = has_nontrivial_input(atom)
            eff_arity = int(effective_arity(atom))
            resta = None
            restm = None

            if eff_arity <= 1:
                cand_list = []
                y_mad_local = 1.0
            else:
                # Extract residual refit leaf when available (multi-atom, cleaner signal)
                refit_leaf = None
                if residual_refit_model is not None:
                    refit_leaf = residual_refit_model.leaf[0]

                refit_note = " (using residual re-fit model)" if refit_leaf is not None else ""
                print(
                    "Checking separability of {}(y) in input space with precision {}{}".format(
                        y_op_str, precision, refit_note
                    )
                )

                if is_multi and models_multi is not None:
                    cand_lists_all = []
                    resta_all = []
                    restm_all = []
                    y_mads_all = []
                    for di, m_i in enumerate(models_multi):
                        tag_to_leaf_i = _build_tag_to_leaf_map(current_ast, m_i)
                        leaf_i = refit_leaf if refit_leaf is not None else tag_to_leaf_i.get(atom.tag)
                        cand_i, resta_i, restm_i, y_mad_i = _check_separability_in_input_space(
                            model=m_i,
                            atom=atom,
                            leaf=leaf_i,
                            datagen_train=train_loaders_all[di],
                            device=device,
                            dtype=dtype,
                            precision_sum=precision,
                            precision_mult=precision,
                            very_verbose=verbose_sep,
                        )
                        cand_lists_all.append(cand_i)
                        if resta_i:
                            resta_all.extend(list(resta_i))
                        if restm_i:
                            restm_all.extend(list(restm_i))
                        y_mads_all.append(float(y_mad_i))
                    cand_list = _merge_sep_candidates(cand_lists_all)
                    resta = sorted(set(resta_all)) if resta_all else None
                    restm = sorted(set(restm_all)) if restm_all else None
                    y_mad_local = _aggregate_losses(y_mads_all, mode=agg_mode, weights=agg_weights)
                else:
                    leaf_to_use = refit_leaf if refit_leaf is not None else tag_to_leaf.get(atom.tag)
                    cand_list, resta, restm, y_mad_local = _check_separability_in_input_space(
                        model=model,
                        atom=atom,
                        leaf=leaf_to_use,
                        datagen_train=datagen_train_noshuffle,
                        device=device,
                        dtype=dtype,
                        precision_sum=precision,
                        precision_mult=precision,
                        very_verbose=verbose_sep,
                    )

            # ----------------------------------------------------------
            # Stage A frugal mode: if we deliberately chose to try disjoint
            # separability first (instead of long composites), restrict the
            # candidate list to disjoint covers (optionally singleton-only)
            # and (optionally) only train the best one(s).
            # ----------------------------------------------------------
            if (sep_filter_mode in ("singleton", "disjoint")) and (not atom_has_compound) and cand_list:
                symb_set = set(getattr(atom, "var_idxs", ()))
                disjoint_cands = []
                overlap_cands = []
                for c in cand_list:
                    g1 = c[1] if len(c) > 1 else []
                    g2 = c[2] if len(c) > 2 else []
                    if sep_filter_mode == "singleton":
                        ok = _is_singleton_disjoint_cover(g1, g2, symb_set)
                    else:
                        ok = _is_clean_disjoint_cover(g1, g2, symb_set)
                    if ok:
                        disjoint_cands.append(c)
                    else:
                        overlap_cands.append(c)

                if disjoint_cands:
                    # Sort by metric when available (lower is better).
                    if any(_candidate_metric(c) is not None for c in disjoint_cands):
                        disjoint_cands.sort(
                            key=lambda c: (
                                _candidate_metric(c)
                                if _candidate_metric(c) is not None
                                else float("inf")
                            )
                        )

                    max_keep = int(getattr(search_hp, "stageA_disjoint_sep_max_candidates", 1))
                    orig_n = len(cand_list)
                    if max_keep > 0:
                        # Disjoint first, then overlapping as fallback
                        cand_list = disjoint_cands[:max_keep] + overlap_cands
                    print(
                        f"[Stage A] disjoint-separability priority ({sep_filter_mode}): "
                        f"trying {min(max_keep, len(disjoint_cands))} disjoint + "
                        f"{len(overlap_cands)} overlapping fallback candidate(s) "
                        f"(from {orig_n} total)."
                    )

            accepted = False
            if (
                compound_exhaustion_deferred
                and enable_compound_global
                and eff_arity > 1
                and str(getattr(atom, "kind", "")).lower() == "nn"
            ):
                (
                    acc_c,
                    new_model_c,
                    new_ast_c,
                    new_loss_c,
                    full_c,
                    _enables_sep_c,
                ) = _try_stageA_compound_during_sep_for_atom(
                    model=model,
                    current_ast=current_ast,
                    atom=atom,
                    tag_to_leaf=tag_to_leaf,
                    datagen_train_noshuffle=datagen_train_noshuffle,
                    datagen_val_noshuffle=datagen_val_noshuffle,
                    device=device,
                    dtype=dtype,
                    leaf_builder=leaf_builder,
                    dual_layer_used=dual_layer_used,
                    search_hp=search_hp,
                    lm_hp=lm_hp,
                    loss_target_eff=loss_target_eff,
                    accept_threshold_eff_cand=accept_threshold_eff_cand,
                    best_val_loss=best_val_loss,
                    current_val_loss=current_val_loss,
                    stageA_under_protest=bool(under_protest),
                    best_train_loss=best_train_loss_initial,
                    loss_scale=loss_scale,
                    model_sep_output=model_sep_output,
                    y_op=y_op,
                    y_op_inv=y_op_inv,
                    Nxvars=Nxvars,
                    x_transform_map=x_transform_map,
                    trig_spec=trig_spec,
                    scale_specs=scale_specs,
                    invariance_feats=invariance_feats,
                    trig_axis_specs_all=trig_axis_specs_all,
                    units_spec=units_spec,
                    enforce_units=bool(enforce_units),
                    units_reject_cb=_units_reject,
                )
                if acc_c:
                    _ast_before = ast_to_human_readable(current_ast)
                    _ast_after = ast_to_human_readable(new_ast_c)
                    _move_parent_model = model
                    _move_parent_models_multi = (
                        list(models_multi) if models_multi is not None else None
                    )
                    _move_parent_val_losses = (
                        list(current_val_losses) if current_val_losses is not None else None
                    )
                    _move_parent_ast = current_ast
                    _move_parent_loss = current_val_loss
                    model = new_model_c
                    current_ast = new_ast_c
                    current_val_loss = new_loss_c
                    if _ast_after == _ast_before:
                        print(
                            "[Stage A Compound] Accepted in-pass candidate kept AST unchanged; "
                            "ignoring no-op accept."
                        )
                    else:
                        _move_details = _stageA_compound_move_details(new_model_c, full_c)
                        _stageA_record_move(
                            move_kind="in_pass_compound",
                            parent_ast=_move_parent_ast,
                            candidate_ast=current_ast,
                            parent_loss=_move_parent_loss,
                            candidate_loss=new_loss_c,
                            reason="in-pass compound candidate accepted",
                            risk_tags={"compound_coordinate"},
                            details=_move_details,
                        )
                        _stageA_begin_pending_full_refit_transaction(
                            move_kind="in_pass_compound",
                            parent_ast=_move_parent_ast,
                            parent_model=_move_parent_model,
                            parent_models_multi=_move_parent_models_multi,
                            parent_val_loss=_move_parent_loss,
                            parent_val_losses=_move_parent_val_losses,
                            candidate_ast=current_ast,
                            candidate_loss=new_loss_c,
                            details=_move_details,
                        )
                        _stageA_sync_shadow_registry(search_hp, current_ast, reason="in-pass compound")
                        changed = True
                        separability_success = True
                        accepted = True
                        (
                            term_ok,
                            term_model,
                            term_ast,
                            term_loss,
                            _term_label,
                        ) = _try_stageA_terminal_closure_probe(
                            model=model,
                            current_ast=current_ast,
                            current_val_loss=float(current_val_loss),
                            datagen_train_noshuffle=datagen_train_noshuffle,
                            datagen_val_noshuffle=datagen_val_noshuffle,
                            device=device,
                            dtype=dtype,
                            search_hp=search_hp,
                            lm_hp=lm_hp,
                            loss_target_eff=loss_target_eff,
                            loss_scale=loss_scale,
                            model_sep_output=model_sep_output,
                            y_op=y_op,
                            y_op_inv=y_op_inv,
                            Nxvars=Nxvars,
                            dual_layer_used=dual_layer_used,
                            x_transform_map=x_transform_map,
                            units_spec=units_spec,
                            enforce_units=bool(enforce_units),
                        )
                        if term_ok:
                            _move_parent_ast = current_ast
                            _move_parent_loss = current_val_loss
                            model = term_model
                            current_ast = term_ast
                            current_val_loss = term_loss
                            _stageA_record_move(
                                move_kind="terminal_closure_after_in_pass_compound",
                                parent_ast=_move_parent_ast,
                                candidate_ast=current_ast,
                                parent_loss=_move_parent_loss,
                                candidate_loss=term_loss,
                                reason="terminal closure accepted after in-pass compound",
                                risk_tags={"terminal_closure"},
                                details={"label": str(_term_label)},
                            )
                        if full_c:
                            print(
                                "[Stage A Compound] Full-variable compound compressed; "
                                "outer map still unresolved"
                            )
                if accepted:
                    break

            # Iterate over candidate splits
            for candidate_sep in cand_list:
                # Decode split type and whether it's overlapping
                op = candidate_sep[0] if candidate_sep else None
                g1 = candidate_sep[1] if len(candidate_sep) > 1 else []
                g2 = candidate_sep[2] if len(candidate_sep) > 2 else []
                cand_metric = _candidate_metric(candidate_sep)
                cand_sep_score = _sep_metric_to_score(cand_metric, precision)
                stagea_sep_candidates_seen += 1
                if cand_metric is not None and math.isfinite(float(cand_metric)):
                    if (stagea_best_sep_metric is None) or (float(cand_metric) < float(stagea_best_sep_metric)):
                        stagea_best_sep_metric = float(cand_metric)
                if cand_sep_score > stagea_best_sep_score:
                    stagea_best_sep_score = float(cand_sep_score)
                # Structural priors from proposal quality alone; confirmed fit quality can raise this later.
                stagea_best_split_score = max(stagea_best_split_score, float(0.6 * cand_sep_score))
                # Extract offset info (4th element for multiplicative splits)
                offset_info = candidate_sep[3] if len(candidate_sep) > 3 else None
                # "Overlapping" means some variable appears in both factors;
                # these are the genuinely partial/ambiguous separations.
                has_overlap = bool(set(g1) & set(g2))
                n_overlap = len(set(g1) & set(g2))

                # ── Dimensional feasibility gate ──────────────────────
                if bool(enforce_units) and units_spec is not None:
                    try:
                        from nestynet_sr.sr_core.units import check_split_feasibility, infer_atom_output_dim, eval_analytic_expr_dim

                        _op_label = "add" if op is torch.add else "mul"

                        # If we're splitting a nested atom, its output units may be constrained by
                        # the surrounding AST. Only apply this pruning gate when we can infer a
                        # unique target dimension for this atom.
                        _target_dim = None
                        if atom is current_ast:
                            _target_dim = units_spec.y_phi_dim
                        else:
                            try:
                                _target_dim = infer_atom_output_dim(current_ast, atom, units_spec)
                            except Exception:
                                _target_dim = None

                        if _target_dim is not None:
                            # Compute compound variable dimension if present
                            _compound_dims = None
                            if atom_has_compound:
                                try:
                                    _cd = {}
                                    _all_inp = get_input_exprs(atom)
                                    _n_cpd = sum(1 for inp in _all_inp if not is_trivial_input(inp))
                                    _zi = 0
                                    for inp in _all_inp:
                                        if not is_trivial_input(inp):
                                            tok = _COMPOUND_Z_TOKEN if _n_cpd == 1 else f"z{_zi}"
                                            _dim = eval_analytic_expr_dim(inp, units_spec.x_dims)
                                            if _dim is not None:
                                                _cd[tok] = _dim
                                            _zi += 1
                                    if _cd:
                                        _compound_dims = _cd
                                except Exception:
                                    pass

                            _feas, _reason = check_split_feasibility(
                                _op_label,
                                g1,
                                g2,
                                _target_dim,
                                units_spec.x_dims,
                                units_spec.unit_system,
                                free_const_dims=units_spec.free_const_dims,
                                fixed_const_dims=getattr(units_spec, "fixed_const_dims", {}),
                                has_offset=False,
                                compound_dims=_compound_dims,
                            )
                            if not _feas:
                                print(f"[Units] Skipping split {g1}/{g2} ({_op_label}): {_reason}")
                                continue
                    except Exception as e:
                        print(f"[Units] Split feasibility check error: {e}")

                # Build list of AST variants to try
                parent_num_segments = atom.kwargs.get("num_segments", search_hp.num_segments_map[dual_layer_used])
                parent_dual_layer = atom.kwargs.get("dual_layer", dual_layer_used)

                # Always try the basic version (status quo)
                basic_subtree = _separability_proposal_to_ast_unified(
                    op, g1, g2, atom, parent_num_segments, parent_dual_layer, parent_tag=atom.tag
                )
                ast_variants = [("no_offset", basic_subtree, None)]

                # If multiplicative with strong offset, also try offset version
                # (skip for compound parents / compound-space splits; offset builder assumes x-space groups)
                if (not atom_has_compound) and op in (torch.mul, torch.multiply) and offset_info and offset_info.strong:
                    from nestynet_sr.sr_core.bridges import separability_proposal_to_ast_with_offset

                    # Compute scaling from y_sub to atom units
                    # y_sub ≈ α * u + β where u is the atom's own output
                    alpha, beta = _compute_atom_scale(
                        model, current_ast, datagen_train_noshuffle,
                        atom.tag, device, dtype
                    )

                    # Convert offset from y_sub units to atom units
                    C_y = offset_info.b_hat * y_mad_local  # offset in y_sub units
                    if abs(alpha) > 1e-10:
                        b_atom = (C_y - beta) / alpha  # offset in atom units
                    else:
                        b_atom = C_y  # fallback if alpha near zero

                    # Under enforce_units, avoid introducing hidden unitful offset constants.
                    # We only add this variant when the offset constant can be represented
                    # either as a dimensionless scale (if output is dimless) or as a *declared*
                    # unitful free constant.
                    b_kind = "free_const"
                    b_name = None
                    b_scope = None
                    add_offset_variant = True
                    if bool(enforce_units) and units_spec is not None:
                        try:
                            from nestynet_sr.sr_core.units import (
                                infer_atom_output_dim,
                            )
                            from nestynet_sr.sr_core.constants import unit_aware_scalar_choice
                            _tgt = units_spec.y_phi_dim if (atom is current_ast) else infer_atom_output_dim(current_ast, atom, units_spec)
                        except Exception:
                            _tgt = None

                        if _tgt is None:
                            add_offset_variant = False
                            print("[Units] Skipping offset variant: could not infer atom output units for offset.")
                        else:
                            _choice = unit_aware_scalar_choice(
                                _tgt,
                                units_spec,
                                prefer_scope="experiment",
                            )
                            if _choice is None:
                                add_offset_variant = False
                                try:
                                    _ds = units_spec.unit_system.format_dim(_tgt)
                                except Exception:
                                    _ds = str(_tgt)
                                print(f"[Units] Skipping offset variant: no declared free_const matches dim {_ds}.")
                            else:
                                b_kind = str(_choice.get("kind", "scale"))
                                b_name = _choice.get("name", None)
                                b_scope = _choice.get("scope", None)

                    if add_offset_variant:
                        offset_subtree = separability_proposal_to_ast_with_offset(
                            g1, g2, parent_num_segments, parent_dual_layer,
                            parent_tag=atom.tag, offset_info=offset_info, y_mad=y_mad_local,
                            b_atom=b_atom,  # pass converted offset in atom units
                            b_kind=b_kind,
                            b_name=b_name,
                            b_scope=b_scope,
                        )
                        offset_value = b_atom  # use atom-scale offset for teacher init
                        ast_variants.append(("with_offset", offset_subtree, offset_value))
                        print(f"[Offset] Strong offset detected: b_hat={offset_info.b_hat:.4f} "
                              f"(y_sub units={C_y:.4f}, atom units={b_atom:.4f}, α={alpha:.3f}, β={beta:.3f})")

                for variant_name, proposed_subtree, offset_value in ast_variants:
                    if accepted:
                        break  # Already accepted a variant

                    # ----------------------------------------------------------
                    # Units consistency gate (Stage A separability variants)
                    #
                    # Use the global units solver to reject dimensionally
                    # impossible split proposals before any fitting.
                    # ----------------------------------------------------------
                    if bool(enforce_units) and (units_spec is not None):
                        try:
                            from nestynet_sr.sr_core.units import check_units_ast

                            _candidate_ast_units = replace_atom_in_ast(
                                current_ast, atom, proposed_subtree
                            )
                            ures = check_units_ast(_candidate_ast_units, units_spec)
                            if not bool(getattr(ures, "ok", False)):
                                reason = getattr(ures, "reason", "unit check failed")
                                print(
                                    f"[Units] Skipping separability variant '{variant_name}' due to units: {reason}"
                                )
                                _units_reject("separability_variant", reason)
                                continue
                        except Exception as e:
                            print(
                                f"[Units] Skipping separability variant '{variant_name}' due to units error: {e}"
                            )
                            _units_reject("separability_variant", e)
                            continue

                    for trial in range(2 * search_hp.ntrial):
                        if (y_op is not None) and (trial >= search_hp.ntrial):
                            break

                        trial_model_size_target = (
                            trial % search_hp.ntrial + 1
                        ) * model_hp.model_size_target

                        # Build candidate AST by replacing atom with separability proposal
                        candidate_ast = replace_atom_in_ast(current_ast, atom, proposed_subtree)

                        def _build_sep_candidate_models(num_segments_candidate: int, *, reuse_current_leaves: bool):
                            # Reusing leaves preserves their original segment counts.  When we
                            # lower segments to satisfy nparam_max, rebuild the whole temporary
                            # candidate so the budget reduction actually takes effect.
                            reuse_map_raw = _build_tag_to_leaf_map(current_ast, model) if reuse_current_leaves else {}
                            reuse_leaves = _clone_reuse_leaves(reuse_map_raw, device, dtype) if reuse_current_leaves else {}
                            temp_model_0, _, cand_ast_updated_0 = build_composite_ast(
                                candidate_ast,
                                num_segments_candidate,
                                dual_layer=parent_dual_layer,
                                leaf_builder=leaf_builder,
                                device=device,
                                dtype=dtype,
                                reuse_leaves=reuse_leaves,
                                freeze_non_nn=freeze_non_nn,
                            )
                            temp_model_0 = _apply_fit_link_to_model(temp_model_0, lm_hp)
                            temp_models_out = [temp_model_0]
                            if is_multi:
                                base_models = (
                                    list(models_multi)
                                    if (models_multi is not None and len(models_multi) == len(train_loaders_all))
                                    else [model for _ in range(len(train_loaders_all))]
                                )
                                for di in range(1, len(train_loaders_all)):
                                    if reuse_current_leaves:
                                        base_m_i = base_models[di] if di < len(base_models) else model
                                        reuse_map_raw_i = _build_tag_to_leaf_map(current_ast, base_m_i)
                                        reuse_leaves_i = _clone_reuse_leaves(reuse_map_raw_i, device, dtype)
                                    else:
                                        reuse_leaves_i = {}
                                    m_i, _, _ = build_composite_ast(
                                        cand_ast_updated_0,
                                        num_segments_candidate,
                                        dual_layer=parent_dual_layer,
                                        leaf_builder=leaf_builder,
                                        device=device,
                                        dtype=dtype,
                                        reuse_leaves=reuse_leaves_i,
                                        freeze_non_nn=freeze_non_nn,
                                    )
                                    m_i = _apply_fit_link_to_model(m_i, lm_hp)
                                    temp_models_out.append(m_i)
                            return temp_model_0, temp_models_out, cand_ast_updated_0

                        # Use the parent's segment count initially for consistency.  If the
                        # candidate is over budget, retry with a smaller whole-candidate
                        # segment count before any teacher initialization or LM fit.
                        num_segments_candidate = int(parent_num_segments)
                        print("Setting number of segments: {} (variant: {})".format(num_segments_candidate, variant_name))
                        preflight_model, _, preflight_ast_updated = build_composite_ast(
                            candidate_ast,
                            num_segments_candidate,
                            dual_layer=parent_dual_layer,
                            leaf_builder=leaf_builder,
                            device=device,
                            dtype=dtype,
                            reuse_leaves={},
                            freeze_non_nn=freeze_non_nn,
                        )
                        preflight_params = int(preflight_model.num_parameters())
                        del preflight_model
                        if model_hp.nparam_max is not None:
                            cap = int(model_hp.nparam_max)
                            temp_params = preflight_params
                            if temp_params > cap:
                                min_segments = max(1, int(getattr(model_hp, "num_segments_min", 1)))
                                print(
                                    "[nparam cap] Candidate model has {} parameters exceeding nparam_max {}; "
                                    "retrying without fitted-leaf reuse at reduced whole-candidate segment count.".format(
                                        temp_params, cap
                                    )
                                )
                                while temp_params > cap and num_segments_candidate > min_segments:
                                    est_segments = int(math.floor(num_segments_candidate * cap / max(float(temp_params), 1.0)))
                                    next_segments = max(min_segments, min(num_segments_candidate - 1, est_segments))
                                    if next_segments >= num_segments_candidate:
                                        next_segments = num_segments_candidate - 1
                                    num_segments_candidate = max(min_segments, int(next_segments))
                                    print(
                                        "[nparam cap] Retrying separability candidate at num_segments={} "
                                        "(previous params={}).".format(num_segments_candidate, temp_params)
                                    )
                                    temp_model, temp_models, candidate_ast_updated = _build_sep_candidate_models(
                                        num_segments_candidate,
                                        reuse_current_leaves=False,
                                    )
                                    temp_params = int(temp_model.num_parameters())
                                if temp_params > cap:
                                    print(
                                        "[nparam cap] Skipping separability candidate: {} parameters still exceed "
                                        "nparam_max {} at num_segments={}.".format(
                                            temp_params, cap, num_segments_candidate
                                        )
                                    )
                                    continue
                                print(
                                    "[nparam cap] Using reduced separability candidate: num_segments={}, "
                                    "params={} <= {}.".format(num_segments_candidate, temp_params, cap)
                                )
                            else:
                                temp_model, temp_models, candidate_ast_updated = _build_sep_candidate_models(
                                    num_segments_candidate,
                                    reuse_current_leaves=True,
                                )
                        else:
                            candidate_ast_updated = preflight_ast_updated
                            temp_model, temp_models, candidate_ast_updated = _build_sep_candidate_models(
                                num_segments_candidate,
                                reuse_current_leaves=True,
                            )

                        # Apply teacher initialization for multiplicative splits
                        # This uses the parent NN to initialize the child NNs with meaningful profiles
                        if op in (torch.mul, torch.multiply):
                            _teacher_init_multiplicative(
                                temp_model, candidate_ast_updated, model, current_ast,
                                datagen_train_noshuffle, device, dtype,
                                parent_tag=atom.tag,
                                offset_value=offset_value
                            )
                        # Apply teacher initialization for additive splits
                        elif op in (torch.add,):
                            _teacher_init_additive(
                                temp_model, candidate_ast_updated, model, current_ast,
                                datagen_train_noshuffle, device, dtype,
                                parent_tag=atom.tag
                            )

                        # Acceptance threshold: shared loss-budget policy.
                        #
                        # Separability candidates are allowed a temporary loss regression
                        # in proportion to the simplification they offer (reduced multivariate
                        # NN complexity / parameter count). We keep the historical
                        # (partial_)max_worsening_factor as a hard ceiling, with the tighter
                        # value used for overlapping splits.
                        max_worsening_factor = float(getattr(search_hp, "max_worsening_factor", 100.0))
                        worsening_floor = float(getattr(search_hp, "worsening_floor", 1.0e-6)) * loss_scale
                        if n_overlap == 0:
                            cap_factor = float(max_worsening_factor)        # disjoint: 100
                        elif n_overlap == 1:
                            cap_factor = float(max_worsening_factor) / 2.0  # almost disjoint: 50
                        else:
                            cap_factor = float(partial_max_worsening_factor) # 2+ shared vars: 3

                        acceptance_noise_floor_raw = _resolve_acceptance_noise_floor_raw(lm_hp, loss_scale)
                        hard_ceiling_sep = _loss_excess_above_floor(global_ceil, acceptance_noise_floor_raw)
                        accept_threshold = _compute_accept_threshold(
                            base_loss=best_val_loss,
                            best_loss=best_val_loss,
                            base_ast=current_ast,
                            cand_ast=candidate_ast_updated,
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
                            is_separability=True,
                            is_partial_separability=bool(has_overlap),
                            max_worsening_factor=float(cap_factor),
                            worsening_floor=float(worsening_floor),
                            hard_ceiling=hard_ceiling_sep,
                            noise_floor=float(acceptance_noise_floor_raw),
                        )
                        (
                            structural_accept_threshold,
                            structural_split_simplification,
                        ) = _accept_threshold_with_structural_target(
                            base_ast=current_ast,
                            cand_ast=candidate_ast_updated,
                            accept_threshold=accept_threshold,
                            loss_target_eff=loss_target_eff,
                        )
                        (
                            structural_accept_threshold,
                            under_protest_cap_sep,
                        ) = _stageA_under_protest_threshold_cap(
                            accept_threshold=structural_accept_threshold,
                            current_val_loss=current_val_loss if current_val_loss is not None else best_val_loss,
                            loss_floor=loss_target_eff,
                            noise_floor=acceptance_noise_floor_raw,
                            under_protest=bool(under_protest),
                            label=f"separability {variant_name}",
                        )
                        if under_protest_cap_sep and structural_accept_threshold < float(accept_threshold):
                            accept_threshold = float(structural_accept_threshold)

                        # Show what split is being attempted
                        proposed_str = ast_to_human_readable(candidate_ast_updated)
                        print(
                            "***Training candidate separable model (variant: {})***, "
                            "acceptance threshold (raw) {:.4e} (base {:.4e})\n"
                            "    Proposed: {}".format(
                                variant_name, accept_threshold, search_hp.loss_acceptable,
                                proposed_str
                            )
                        )
                        if structural_accept_threshold > float(accept_threshold):
                            print(
                                "[Stage A] Structural split target enabled: "
                                f"arity signature {_nn_split_signature(current_ast)}"
                                f" → {_nn_split_signature(candidate_ast_updated)}, "
                                f"target-quality threshold {structural_accept_threshold:.4e}"
                            )
                        if under_protest_cap_sep:
                            print("[Stage A] Under-protest branch: requiring non-regressing validation loss.")
                        overlap_metric = None
                        if (
                            has_overlap
                            and len(temp_models) == 1
                            and op in (torch.add, torch.mul, torch.multiply)
                            and bool(getattr(search_hp, "overlap_truth_screen_enable", True))
                        ):
                            overlap_truth_model = residual_refit_model if residual_refit_model is not None else model
                            overlap_truth_ast = residual_refit_ast if residual_refit_ast is not None else current_ast
                            overlap_metric = _evaluate_overlap_truth_metric(
                                parent_model=overlap_truth_model,
                                current_ast=overlap_truth_ast,
                                parent_tag=atom.tag,
                                g1=g1,
                                g2=g2,
                                datagen=datagen_train_noshuffle,
                                device=device,
                                dtype=dtype,
                                op=op,
                                offset_value=offset_value,
                                max_batches=int(getattr(search_hp, "overlap_truth_max_batches", 4)),
                                anchor_rel_eps=float(getattr(search_hp, "overlap_truth_anchor_rel_eps", 1.0e-8)),
                            )
                            if overlap_metric is None:
                                print(
                                    "[Overlap truth] Screen unavailable for this candidate; "
                                    "proceeding with ordinary LM fit."
                                )
                            else:
                                overlap_ok, overlap_tol = _overlap_truth_metric_is_acceptable(
                                    overlap_metric,
                                    op=op,
                                    precision=precision,
                                    search_hp=search_hp,
                                )
                                print(
                                    "[Overlap truth] normalized RMS {:.3e}, peak {:.3e}, "
                                    "valid {:.1%} (tol {:.3e})".format(
                                        float(overlap_metric["normalized_rms"]),
                                        float(overlap_metric["normalized_peak"]),
                                        float(overlap_metric["valid_fraction"]),
                                        float(overlap_tol),
                                    )
                                )
                                if not overlap_ok:
                                    print(
                                        f"{YELLOW}[Overlap truth] Rejecting overlap candidate before LM: "
                                        f"function-space residual exceeds tolerance.{RESET}"
                                    )
                                    continue

                        (
                            accepted_candidate,
                            best_val_loss_candidate,
                            best_train_loss_candidate,
                            per_val_losses_candidate,
                            _per_train_losses_candidate,
                            temp_models,
                        ) = _train_candidate_models(
                            temp_models,
                            structural_accept_threshold,
                            extra_train_factories=None,
                        )

                        if accepted_candidate:
                            structural_target_accept = bool(
                                structural_split_simplification
                                and best_val_loss_candidate > float(accept_threshold)
                                and best_val_loss_candidate <= float(structural_accept_threshold)
                            )
                            if structural_target_accept:
                                print(
                                    f"{GREEN}Accepted{RESET} structural split at target-quality loss "
                                    f"{_loss_str(best_val_loss_candidate, lm_hp)} "
                                    f"(strict threshold {accept_threshold:.4e})"
                                )
                            fit_ratio = 0.0
                            try:
                                fit_ratio = float(accept_threshold) / max(
                                    float(best_val_loss_candidate), 1.0e-30
                                )
                            except Exception:
                                fit_ratio = 0.0
                            if not math.isfinite(fit_ratio) or fit_ratio < 0.0:
                                fit_ratio = 0.0
                            fit_score = float(fit_ratio / (1.0 + fit_ratio))
                            split_score_candidate = float(
                                max(
                                    cand_sep_score,
                                    0.6 * cand_sep_score + 0.4 * fit_score,
                                )
                            )
                            stagea_best_split_score = max(
                                stagea_best_split_score, split_score_candidate
                            )
                            stagea_split_accept_count += 1
                            # Training loss sanity check (same as Early Compound)
                            max_train_degradation = float(getattr(search_hp, "max_train_degradation", 100.0))
                            passes_relative = (
                                best_train_loss_initial is None
                                or best_train_loss_initial <= 0
                                or best_train_loss_candidate <= max_train_degradation * best_train_loss_initial
                            )
                            passes_absolute = best_train_loss_candidate <= loss_target_eff

                            if not passes_relative and not passes_absolute:
                                degradation = best_train_loss_candidate / best_train_loss_initial if best_train_loss_initial else float('inf')
                                print(
                                    f"{RED}Rejected{RESET} separation (variant: {variant_name}): "
                                    f"training loss {degradation:.0f}× worse than current model"
                                )
                                continue  # Skip to next trial

                            try:
                                noisy_overlap_n_eff = getattr(lm_hp, "acceptance_noise_n_eff", None)
                                if noisy_overlap_n_eff is None:
                                    noisy_overlap_n_eff = _loader_n_eff(datagen_val_noshuffle)
                            except Exception:
                                noisy_overlap_n_eff = None
                            split_kind_label = "mul" if op in (torch.mul, torch.multiply) else "add"
                            noisy_overlap_ok, noisy_overlap_reason = _stageA_noisy_overlap_split_gate(
                                split_kind=split_kind_label,
                                has_overlap=bool(has_overlap),
                                base_ast=current_ast,
                                cand_ast=candidate_ast_updated,
                                base_val_loss=best_val_loss,
                                cand_val_loss=best_val_loss_candidate,
                                noise_floor=acceptance_noise_floor_raw,
                                n_eff=noisy_overlap_n_eff,
                            )
                            if not noisy_overlap_ok:
                                print(f"{YELLOW}[Overlap gauge] {noisy_overlap_reason}{RESET}")
                                continue

                            coe_overlap_ok, coe_overlap_reason, coe_overlap_summary = _stageA_overlap_split_committee_gate(
                                base_model=model,
                                cand_model=temp_models[0],
                                split_kind=split_kind_label,
                                has_overlap=bool(has_overlap),
                                base_val_loss=best_val_loss,
                                cand_val_loss=best_val_loss_candidate,
                                noise_floor=acceptance_noise_floor_raw,
                                under_protest=bool(under_protest),
                                lm_hp=lm_hp,
                                y_op=y_op,
                                y_op_inv=y_op_inv,
                                dtype=dtype,
                                device=device,
                                data_hp=data_hp,
                                split_diagnostic=overlap_metric,
                                structural_simplification=bool(structural_split_simplification),
                                structural_budget_multiplier=_stageA_loss_budget_multiplier(
                                    base_loss=best_val_loss,
                                    allowed_loss=structural_accept_threshold,
                                    noise_floor=acceptance_noise_floor_raw,
                                ),
                            )
                            if isinstance(coe_overlap_summary, dict) and coe_overlap_summary.get("gate_status") in {"evaluated", "accepted", "accepted_provisional", "veto"}:
                                print(_format_stageA_overlap_split_committee_report(coe_overlap_summary))
                            if not coe_overlap_ok:
                                print(f"{YELLOW}[CoE Stage A] {coe_overlap_reason}{RESET}")
                                continue

                            separability_success = True
                            rest_add = (
                                resta if rest_add is None else (rest_add + resta if resta else rest_add)
                            )
                            rest_mult = (
                                restm
                                if rest_mult is None
                                else (rest_mult + restm if restm else rest_mult)
                            )
                            print(
                                f"{GREEN}Accepted{RESET} separation (variant: {variant_name}), val-loss {_loss_str(best_val_loss_candidate, lm_hp)}"
                            )

                            # Show current expression structure
                            try:
                                expr_str = _compact_expression_repr(
                                    candidate_ast, max_length=240, y_op_inv=y_op_inv
                                )
                                print(f"[Stage A]   Current: {expr_str}")
                            except Exception as e:
                                print(f"[Stage A]   Expression display error: {e}")
                                # Fallback: try without y_op_inv
                                try:
                                    expr_str = _compact_expression_repr(
                                        candidate_ast, max_length=240, y_op_inv=None
                                    )
                                    print(f"[Stage A]   Current: {expr_str}")
                                except Exception:
                                    pass

                            dual_layer_used = parent_dual_layer
                            torch.save(
                                dict(
                                    y_op=y_op,
                                    y_op_inv=y_op_inv,
                                    Nxvars=Nxvars,
                                    # num_segments is now stored per-atom in the AST
                                    dual_layer=dual_layer_used,
                                    x_transform=x_transform_map,
                                    model_state_dict=temp_model.state_dict(),
                                    ast=candidate_ast_updated,  # Save updated AST with correct per-atom kwargs
                                    val_loss=best_val_loss_candidate,  # Track fit quality for diagnostics
                                    val_losses=list(per_val_losses_candidate),
                                    val_loss_agg_mode=str(agg_mode),
                                    val_loss_agg_weights=list(agg_weights) if agg_weights is not None else None,
                                    dataset_ids=list(dataset_ids) if dataset_ids is not None else None,
                                    fit_y_link=getattr(lm_hp, "fit_y_link", None),
                                    fit_y_link_scale=float(getattr(lm_hp, "fit_y_link_scale", 1.0)),
                                ),
                                model_sep_output,
                            )

                            # Adopt the candidate as the current model
                            _move_parent_model = model
                            _move_parent_models_multi = (
                                list(models_multi) if models_multi is not None else None
                            )
                            _move_parent_val_losses = (
                                list(current_val_losses)
                                if current_val_losses is not None
                                else None
                            )
                            _move_parent_ast = current_ast
                            _move_parent_loss = current_val_loss
                            model = temp_models[0]
                            models_multi = list(temp_models)
                            current_ast = candidate_ast_updated
                            current_val_loss = best_val_loss_candidate
                            current_val_losses = list(per_val_losses_candidate)
                            _split_risk_tags = {"split_accept"}
                            if bool(has_overlap):
                                _split_risk_tags.add("overlap_split")
                            if bool(has_overlap) and op in (torch.mul, torch.multiply):
                                _split_risk_tags.add("overlap_multiplicative_split")
                            if bool(under_protest):
                                _split_risk_tags.add("under_protest_split")
                            _split_move_details = {
                                "variant": str(variant_name),
                                "op": getattr(op, "__name__", str(op)),
                                "group1": _stageA_split_group_record_payload(g1),
                                "group2": _stageA_split_group_record_payload(g2),
                                "has_overlap": bool(has_overlap),
                                "structural_target_accept": bool(structural_target_accept),
                                "split_score": float(split_score_candidate),
                                "coe_stageA_overlap_split_gate": coe_overlap_summary,
                                "coe_provisional_budget_admission": bool(
                                    isinstance(coe_overlap_summary, dict)
                                    and coe_overlap_summary.get(
                                        "provisional_budget_admission", False
                                    )
                                ),
                            }
                            _stageA_record_move(
                                move_kind="separability_split",
                                parent_ast=_move_parent_ast,
                                candidate_ast=current_ast,
                                parent_loss=_move_parent_loss,
                                candidate_loss=best_val_loss_candidate,
                                reason=f"separability variant {variant_name} accepted",
                                risk_tags=_split_risk_tags,
                                details=_split_move_details,
                            )
                            if _split_move_details["coe_provisional_budget_admission"]:
                                _stageA_begin_pending_full_refit_transaction(
                                    move_kind="separability_split",
                                    parent_ast=_move_parent_ast,
                                    parent_model=_move_parent_model,
                                    parent_models_multi=_move_parent_models_multi,
                                    parent_val_loss=_move_parent_loss,
                                    parent_val_losses=_move_parent_val_losses,
                                    candidate_ast=current_ast,
                                    candidate_loss=best_val_loss_candidate,
                                    details=_split_move_details,
                                )
                            _stageA_sync_shadow_registry(search_hp, current_ast, reason="separation")
                            changed = True
                            any_sep_split = True
                            accepted = True
                            break  # Exit trial loop and restart atom iteration
                        else:
                            print(
                                "Rejected separation (variant: {}), val-loss {:.4e}".format(variant_name, best_val_loss_candidate)
                            )

                    if accepted:
                        break  # Exit variant loop

                if accepted:
                    break  # Exit candidate_sep loop

            # If nothing accepted, mark this atom as done
            if not accepted:
                done_atoms.add(id(atom))
            else:
                # We accepted a candidate, AST has changed, break out of atom loop
                break  # Exit atom loop to restart with new AST

        # After compound or other acceptance, continue separability search on new AST

        # If we deferred compound exhaustion to try separability first, but no move
        # was accepted, fall back to the deferred compound exhaustion now. This
        # prevents the dynamic policy from "painting itself into a corner" when
        # a clean-looking split fails later validation/training.
        if (not changed) and compound_exhaustion_deferred:
            print(
                "[Stage A] Deferred compound exhaustion: separability-first made no progress; "
                "now exhausting compounds."
            )
            compound_exhaustion_loop_active = True
            _compound_prev_ast_repr = ast_to_human_readable(current_ast)
            while compound_exhaustion_loop_active:
                compound_found_this_pass = False
                nn_atoms = collect_nn_atoms(current_ast)
                tag_to_leaf = _build_tag_to_leaf_map(current_ast, model)

                # Extract compound detection parameters (same as above)
                enable_compound = bool(getattr(search_hp, "enable_compound_detection", False)) and (not is_multi)
                compound_max_vars = int(getattr(search_hp, "compound_max_vars", 4))
                compound_max_exponent = int(getattr(search_hp, "compound_max_exponent", 5))
                compound_threshold = float(getattr(search_hp, "compound_threshold", 0.05))
                compound_max_batches = int(getattr(search_hp, "compound_max_batches", 4))
                COMPOUND_CONF_THRESHOLD = float(getattr(search_hp, "compound_confidence_gate", 0.85))

                if not enable_compound:
                    break

                for atom in nn_atoms:
                    if effective_arity(atom) <= 1:
                        acc_m, new_model_m, new_ast_m, new_loss_m = _try_stageA_univariate_monomial_for_atom(
                            model=model,
                            current_ast=current_ast,
                            atom=atom,
                            tag_to_leaf=tag_to_leaf,
                            datagen_train_noshuffle=datagen_train_noshuffle,
                            datagen_val_noshuffle=datagen_val_noshuffle,
                            device=device,
                            dtype=dtype,
                            leaf_builder=leaf_builder,
                            dual_layer_used=dual_layer_used,
                            search_hp=search_hp,
                            lm_hp=lm_hp,
                            loss_target_eff=loss_target_eff,
                            accept_threshold_eff_cand=accept_threshold_eff_cand,
                            best_val_loss=best_val_loss,
                            current_val_loss=current_val_loss,
                            stageA_under_protest=bool(under_protest),
                            best_train_loss=best_train_loss_initial,
                            loss_scale=loss_scale,
                            units_spec=units_spec,
                            enforce_units=bool(enforce_units),
                            units_reject_cb=_units_reject,
                            y_op=y_op,
                            y_op_inv=y_op_inv,
                        )
                        if acc_m:
                            _ast_before = ast_to_human_readable(current_ast)
                            _ast_after = ast_to_human_readable(new_ast_m)
                            _move_parent_ast = current_ast
                            _move_parent_loss = current_val_loss
                            model = new_model_m
                            current_ast = new_ast_m
                            current_val_loss = new_loss_m
                            if _ast_after == _ast_before:
                                print(
                                    "[Stage A Monomial] Accepted candidate kept AST unchanged; "
                                    "ignoring no-op accept."
                                )
                                break
                            _stageA_record_move(
                                move_kind="deferred_stageA_monomial",
                                parent_ast=_move_parent_ast,
                                candidate_ast=current_ast,
                                parent_loss=_move_parent_loss,
                                candidate_loss=new_loss_m,
                                reason="deferred Stage-A monomial candidate accepted",
                                risk_tags={"terminal_closure"} if not collect_nn_atoms(current_ast) else None,
                            )
                            _stageA_sync_shadow_registry(search_hp, current_ast, reason="deferred Stage A monomial")
                            changed = True
                            compound_found_this_pass = True
                            break
                        continue

                    if str(getattr(atom, "kind", "")).lower() != "nn":
                        continue

                    if not (2 <= effective_arity(atom) <= compound_max_vars):
                        continue

                    atom_already_compound = has_nontrivial_input(atom)
                    leaf = tag_to_leaf.get(atom.tag)
                    skip_same_arity_wrappers = False

                    if atom_already_compound:
                        inputs = get_input_exprs(atom)
                        z_expr_cur = inputs[0]
                        extra_var_idxs_cur = [int(inp.var_idxs[0]) for inp in inputs[1:] if is_trivial_input(inp)]
                        extra_input_asts_cur = [inp for inp in inputs[1:] if not is_trivial_input(inp)]
                        if (extra_var_idxs_cur or extra_input_asts_cur) and leaf is not None:
                            already_sep = _quick_separability_check(
                                model=model,
                                leaf=leaf,
                                z_expr=z_expr_cur,
                                extra_var_idxs=extra_var_idxs_cur,
                                extra_input_asts=extra_input_asts_cur,
                                datagen_train=datagen_train_noshuffle,
                                device=device,
                                dtype=dtype,
                            )
                            if already_sep:
                                skip_same_arity_wrappers = True
                                print(
                                    f"[Compound] Existing compound separates from extras {extra_var_idxs_cur}, "
                                    "but checking only extensions that consume the separated coordinate."
                                )
                        print(
                            f"[Variable Extension] Checking iterative extension for compound atom NN{list(atom.var_idxs)}..."
                        )
                    else:
                        print(
                            f"[Variable Extension] Detecting compound structure for NN{list(atom.var_idxs)}..."
                        )

                    try:
                        compound_proposals, compound_oracle_trig_specs = _detect_compound_variable_for_atom(
                            model=model,
                            atom=atom,
                            leaf=leaf,
                            datagen_train=datagen_train_noshuffle,
                            device=device,
                            max_exponent=compound_max_exponent,
                            precision=compound_threshold,
                            max_batches=compound_max_batches,
                            enable_linear=bool(getattr(search_hp, "compound_try_linear", True)),
                            max_linear_coeff=int(getattr(search_hp, "compound_linear_max_coeff", 2)),
                            enable_radial=bool(getattr(search_hp, "compound_try_radial", True)),
                            radial_max_group_size=int(getattr(search_hp, "compound_radial_max_group_size", 3)),
                            radial_cos_threshold=float(getattr(search_hp, "compound_radial_cos_threshold", 0.95)),
                            radial_try_sqrt=bool(getattr(search_hp, "compound_radial_try_sqrt", True)),
                            enable_shift=bool(getattr(search_hp, "compound_try_shift", True)),
                            shift_min_r2=float(getattr(search_hp, "compound_shift_min_r2", 0.85)),
                            shift_min_abs_slope=float(getattr(search_hp, "compound_shift_min_abs_slope", 1e-6)),
                            shift_require_in_range=bool(getattr(search_hp, "compound_shift_require_in_range", True)),
                            shift_max_axes_per_atom=int(getattr(search_hp, "compound_shift_max_axes_per_atom", 2)),
                            scaling_features=scale_specs,
                            invariance_features=invariance_feats,
                            trig_axis_specs=trig_axis_specs_all,
                            enable_mixed_compound=bool(getattr(search_hp, "compound_try_mixed", True)),
                            enable_retained_axis_wrappers=bool(
                                getattr(search_hp, "compound_try_retained_axis_wrappers", True)
                            ),
                            units_spec=units_spec,
                            enforce_units=bool(enforce_units),
                            shadow_registry=_stageA_shadow_registry(search_hp),
                            gs_cfg=getattr(search_hp, "gs_config", None),
                        )
                    except Exception as e:
                        print(f"[Compound Exhaustion] Detection failed: {type(e).__name__}: {e}")
                        compound_proposals = []
                        compound_oracle_trig_specs = []

                    phase_hint_props = _phase_hint_compound_proposals_for_atom(
                        getattr(search_hp, "phase_hints", []),
                        atom,
                        min_confidence=COMPOUND_CONF_THRESHOLD,
                    )
                    if phase_hint_props:
                        compound_proposals = list(compound_proposals or []) + list(phase_hint_props)
                        print(
                            f"[PhaseScan] Added {len(phase_hint_props)} phase carrier(s) "
                            "to Stage-A compound proposals."
                        )

                    compound_proposals = _stageA_append_compound_replay_proposals(
                        compound_proposals or [],
                        search_hp=search_hp,
                        lm_hp=lm_hp,
                        current_ast=current_ast,
                        atom=atom,
                        Nxvars=Nxvars,
                        x_transform_map=x_transform_map,
                        units_spec=units_spec,
                    )
                    compound_proposals = _stageA_append_visible_buckingham_1d_prefactor_proposals(
                        compound_proposals,
                        current_ast=current_ast,
                        atom=atom,
                        units_spec=units_spec,
                        enforce_units=bool(enforce_units),
                        search_hp=search_hp,
                        x_transform_map=x_transform_map,
                    )
                    best_conf = _compound_best_proposal_confidence(compound_proposals)
                    if compound_proposals and best_conf >= COMPOUND_CONF_THRESHOLD:
                        print(
                            f"[Compound Exhaustion] High-confidence proposal (conf={best_conf:.3f}), trying..."
                        )
                        acc_c, new_model, new_ast, new_loss, full_c, _enables_sep = _try_compound_candidates_for_atom(
                            proposals=compound_proposals,
                            model=model,
                            current_ast=current_ast,
                            atom=atom,
                            tag_to_leaf=tag_to_leaf,
                            datagen_train_noshuffle=datagen_train_noshuffle,
                            datagen_val_noshuffle=datagen_val_noshuffle,
                            device=device,
                            dtype=dtype,
                            leaf_builder=leaf_builder,
                            dual_layer_used=dual_layer_used,
                            search_hp=search_hp,
                            lm_hp=lm_hp,
                            loss_target_eff=loss_target_eff,
                            accept_threshold_eff_cand=accept_threshold_eff_cand,
                            best_val_loss=best_val_loss,
                            current_val_loss=current_val_loss,
                            stageA_under_protest=bool(under_protest),
                            best_train_loss=best_train_loss_initial,
                            loss_scale=loss_scale,
                            model_sep_output=model_sep_output,
                            y_op=y_op,
                            y_op_inv=y_op_inv,
                            Nxvars=Nxvars,
                            x_transform_map=x_transform_map,
                            trig_spec=trig_spec,
                            units_spec=units_spec,
                            enforce_units=bool(enforce_units),
                            units_reject_cb=_units_reject,
                            allow_iterative_extension=atom_already_compound,
                            skip_same_arity_if_already_sep=skip_same_arity_wrappers,
                            oracle_trig_specs=compound_oracle_trig_specs,
                            scaling_features=scale_specs,
                        )
                        if acc_c:
                            _ast_before = ast_to_human_readable(current_ast)
                            _ast_after = ast_to_human_readable(new_ast)
                            _move_parent_model = model
                            _move_parent_models_multi = (
                                list(models_multi) if models_multi is not None else None
                            )
                            _move_parent_val_losses = (
                                list(current_val_losses) if current_val_losses is not None else None
                            )
                            _move_parent_ast = current_ast
                            _move_parent_loss = current_val_loss
                            model = new_model
                            current_ast = new_ast
                            current_val_loss = new_loss
                            if _ast_after == _ast_before:
                                print(
                                    "[Compound Exhaustion] Accepted candidate kept AST unchanged; "
                                    "ignoring no-op accept."
                                )
                                break
                            _move_details = _stageA_compound_move_details(new_model, full_c)
                            _stageA_record_move(
                                move_kind="deferred_compound_exhaustion",
                                parent_ast=_move_parent_ast,
                                candidate_ast=current_ast,
                                parent_loss=_move_parent_loss,
                                candidate_loss=new_loss,
                                reason="deferred compound exhaustion candidate accepted",
                                risk_tags={"compound_coordinate"},
                                details=_move_details,
                            )
                            _stageA_begin_pending_full_refit_transaction(
                                move_kind="deferred_compound_exhaustion",
                                parent_ast=_move_parent_ast,
                                parent_model=_move_parent_model,
                                parent_models_multi=_move_parent_models_multi,
                                parent_val_loss=_move_parent_loss,
                                parent_val_losses=_move_parent_val_losses,
                                candidate_ast=current_ast,
                                candidate_loss=new_loss,
                                details=_move_details,
                            )
                            _stageA_sync_shadow_registry(search_hp, current_ast, reason="deferred compound exhaustion")
                            changed = True
                            compound_found_this_pass = True
                            (
                                term_ok,
                                term_model,
                                term_ast,
                                term_loss,
                                _term_label,
                            ) = _try_stageA_terminal_closure_probe(
                                model=model,
                                current_ast=current_ast,
                                current_val_loss=float(current_val_loss),
                                datagen_train_noshuffle=datagen_train_noshuffle,
                                datagen_val_noshuffle=datagen_val_noshuffle,
                                device=device,
                                dtype=dtype,
                                search_hp=search_hp,
                                lm_hp=lm_hp,
                                loss_target_eff=loss_target_eff,
                                loss_scale=loss_scale,
                                model_sep_output=model_sep_output,
                                y_op=y_op,
                                y_op_inv=y_op_inv,
                                Nxvars=Nxvars,
                                dual_layer_used=dual_layer_used,
                                x_transform_map=x_transform_map,
                                units_spec=units_spec,
                                enforce_units=bool(enforce_units),
                            )
                            if term_ok:
                                _move_parent_ast = current_ast
                                _move_parent_loss = current_val_loss
                                model = term_model
                                current_ast = term_ast
                                current_val_loss = term_loss
                                _stageA_record_move(
                                    move_kind="terminal_closure_after_deferred_compound",
                                    parent_ast=_move_parent_ast,
                                    candidate_ast=current_ast,
                                    parent_loss=_move_parent_loss,
                                    candidate_loss=term_loss,
                                    reason="terminal closure accepted after deferred compound exhaustion",
                                    risk_tags={"terminal_closure"},
                                    details={"label": str(_term_label)},
                                )
                                full_compound_solved = True
                            if full_c:
                                print(
                                    "[Compound Exhaustion] Full-variable compound compressed; outer map still unresolved"
                                )
                                full_compound_solved = True
                            break

                if not compound_found_this_pass:
                    break

                # Identical-state guard: if the AST is unchanged after "accepting"
                # a compound, we're stuck (e.g. passthrough re-accepted).  Break.
                _compound_cur_repr = ast_to_human_readable(current_ast)
                if _compound_cur_repr == _compound_prev_ast_repr:
                    print("[Compound Exhaustion] AST unchanged after accept; breaking to avoid infinite loop.")
                    break
                _compound_prev_ast_repr = _compound_cur_repr

        try:
            if changed and not collect_nn_atoms(current_ast):
                print("[Stage A] Terminal analytic closure removed all NN atoms; ending Stage A run.")
                _clear_plain_random_branch_after_structural_accept("terminal analytic closure")
                _refresh_fit_link_original_y_certificate("terminal analytic closure", quiet=True)
                separability_success = True
                break
        except Exception:
            pass

        if changed:
            _clear_plain_random_branch_after_structural_accept("accepted Stage-A rewrite")
            _refresh_fit_link_original_y_certificate("accepted Stage-A rewrite", quiet=True)
            if _x_precond_structural_gate is not None:
                _x_precond_made_progress = True
            i += 1  # Increment iteration counter before restarting
            if _stageA_should_restart("accepted Stage-A rewrite"):
                separable = True  # Continue the separability search with the modified AST
                continue  # Restart the while loop to check for further separability
            separable = False
            break

        if early_made_progress:
            # Early Compound or PureDiff changed the AST — loop back to try
            # separability on the new structure before resorting to x-preconditioning.
            _clear_plain_random_branch_after_structural_accept("early compound/purediff")
            _refresh_fit_link_original_y_certificate("early compound/purediff", quiet=True)
            i += 1
            if _stageA_should_restart("early compound/purediff"):
                separable = True
                continue
            separable = False
            break

        if not changed or len(done_atoms) >= len(nn_atoms):
            # If Stage A is stalling but we still have multivariate leaves, optionally
            # try a small number of x-preconditioning retries (reciprocal / trig).
            if (
                x_precond_enable
                and (mode == "full")
                and (y_op is None)
                and (not under_protest)
                and (x_precond_attempts < x_precond_max_extra_passes)
                and (not any_sep_split)  # skip if an actual add/mult split simplified the AST (compound-only is OK)
            ):
                multivar_axes = _nn_multivar_axes(current_ast)
                # Don't propose per-axis transforms for axes already absorbed
                # into compound expressions (e.g. x4 in x0*x3*x4^{-1})
                multivar_axes = {
                    a for a in multivar_axes
                    if not _axis_is_inside_compound_input(current_ast, a)
                }
                if multivar_axes:
                    did_restart = False

                    # Pass 1: reciprocal transforms suggested by scaling features
                    if (not did_restart) and ("recip" not in x_precond_applied):
                        recip_map = _propose_reciprocal_x_map(
                            scale_specs,
                            multivar_axes=multivar_axes,
                            x_transform_map=x_transform_map,
                            tol=x_precond_scaling_tol,
                        )
                        if recip_map:
                            from .xcoord import XCoordSystem
                            cand_map = dict(x_transform_map)
                            cand_map.update(recip_map)

                            xcoord = XCoordSystem.from_map(cand_map, Nx_raw=Nxvars)

                            # Units consistency: reject x-preconditioning transforms
                            # that violate dimensional rules (e.g. log of a unitful axis).
                            recip_units_ok = True
                            if units_spec is not None and units_raw_x_dims is not None:
                                try:
                                    _ = xcoord.internal_x_dims(units_spec.unit_system, units_raw_x_dims)
                                except Exception as e:
                                    print(
                                        f"[Units] Skipping reciprocal x-preconditioning proposal due to units: {e}"
                                    )
                                    _units_reject("x_precond_reciprocal", e)
                                    x_precond_applied.add("recip")
                                    recip_units_ok = False

                            if recip_units_ok:
                                def x_op_precond(x):
                                    return xcoord.apply_torch(x)

                                cand_train_dl, cand_val_dl = _build_xtransformed_loaders(
                                    dataset_train, dataset_val, data_hp, x_op_precond
                                )

                                if _loader_all_finite(cand_train_dl) and _loader_all_finite(cand_val_dl):
                                    ax_list = ",".join(str(a) for a in sorted(recip_map.keys()))
                                    print(
                                        f"[Stage A] x-preconditioning: retrying with reciprocal"
                                        f" substitutions on axes [{ax_list}]"
                                    )

                                    # Save state BEFORE applying transform (for revert on failure)
                                    x_precond_saved_state = {
                                        "x_transform_map": dict(x_transform_map),
                                        "train_dl": datagen_train_noshuffle,
                                        "val_dl": datagen_val_noshuffle,
                                        "current_ast": current_ast,
                                        "model": model,
                                        "i": i,
                                        "current_val_loss": current_val_loss,
                                        "kind": "recip",
                                        "dual_layer_used": dual_layer_used,
                                        "feats": feats,
                                        "scale_specs": scale_specs,
                                        "parity_specs": parity_specs,
                                        "trig_spec": trig_spec,
                                        "invariance_feats": invariance_feats,
                                    }

                                    x_transform_map = cand_map
                                    _refresh_units_spec_for_xmap(x_transform_map)
                                    datagen_train_noshuffle = cand_train_dl
                                    datagen_val_noshuffle = cand_val_dl

                                    # Force a clean refit in the new x-coordinates
                                    model = None
                                    current_val_loss = None
                                    dual_layer_used = False
                                    under_protest = False

                                    i = 0
                                    x_precond_applied.add("recip")
                                    x_precond_attempts += 1
                                    did_restart = True

                    # Pass 2: trig substitution on a detected trig-like axis
                    # Uses oracle trig_scale_specs (not FFT) for gating and parameters
                    _oracle_trig_for_precond = None
                    if trig_scale_specs:
                        _candidates = [
                            ts for ts in trig_scale_specs
                            if int(ts.axis) in multivar_axes
                            and int(ts.axis) not in x_transform_map
                            and not ts.compound_name  # only trivial (per-axis) targets
                        ]
                        if _candidates:
                            _oracle_trig_for_precond = min(_candidates, key=lambda ts: ts.rel_std)

                    if (
                        (not did_restart)
                        and ("trig" not in x_precond_applied)
                        and (_oracle_trig_for_precond is not None)
                    ):
                        axis = int(_oracle_trig_for_precond.axis)
                        if _axis_is_inside_compound_input(current_ast, axis):
                            print(f"[Stage A] x-preconditioning: axis x{axis} is inside a compound leaf; skipping per-axis trig substitution.")
                            x_precond_applied.add("trig")
                        elif (axis in multivar_axes) and (axis not in x_transform_map):
                            if not _axis_is_coupled_by_invariance(axis, feats):
                                from .xcoord import XCoordSystem

                                omega_detected = _snap_omega(float(_oracle_trig_for_precond.omega))

                                # Build candidate omega list: detected + common fallbacks
                                omega_candidates = []
                                for w in [omega_detected, 1.0, 2.0]:
                                    # Deduplicate (within tolerance)
                                    if not any(abs(w - existing) < 0.05 for existing in omega_candidates):
                                        omega_candidates.append(w)

                                # Get list of already-tried omegas for this axis
                                tried = trig_tried_omegas.get(axis, [])

                                # Find next untried omega
                                omega_to_try = None
                                for w in omega_candidates:
                                    if not any(abs(w - t) < 0.05 for t in tried):
                                        omega_to_try = w
                                        break

                                if omega_to_try is not None:
                                    # Use trig function from oracle (already tested cos vs sin)
                                    trig_fn = _oracle_trig_for_precond.trig_fn

                                    cand_map = dict(x_transform_map)
                                    _precond_pipe = []
                                    if omega_to_try != 1.0:
                                        _precond_pipe.append({"kind": "scale", "scale": omega_to_try})
                                    _precond_pipe.append({"kind": trig_fn})
                                    cand_map[axis] = {
                                        "pipeline": _precond_pipe,
                                        "meta": {"source": "stageA_precond_trig"},
                                    }

                                    xcoord = XCoordSystem.from_map(cand_map, Nx_raw=Nxvars)

                                    # Units consistency: ensure x-preconditioning transform
                                    # is dimensionally valid.
                                    trig_units_ok = True
                                    if units_spec is not None and units_raw_x_dims is not None:
                                        try:
                                            _ = xcoord.internal_x_dims(units_spec.unit_system, units_raw_x_dims)
                                        except Exception as e:
                                            print(
                                                f"[Units] Skipping trig x-preconditioning proposal due to units: {e}"
                                            )
                                            _units_reject("x_precond_trig", e)
                                            # The unit failure is axis-level: changing the numeric
                                            # omega cannot make a unit-bearing trig argument legal.
                                            trig_tried_omegas.setdefault(axis, []).append(omega_to_try)
                                            x_precond_applied.add("trig")
                                            trig_units_ok = False

                                    if trig_units_ok:
                                        def x_op_trig(x):
                                            return xcoord.apply_torch(x)

                                        cand_train_dl, cand_val_dl = _build_xtransformed_loaders(
                                            dataset_train, dataset_val, data_hp, x_op_trig
                                        )

                                        if _loader_all_finite(cand_train_dl) and _loader_all_finite(cand_val_dl):
                                            remaining = len(omega_candidates) - len(tried) - 1
                                            print(
                                                f"[Stage A] x-preconditioning: trying "
                                                f"{trig_fn}({omega_to_try:.4g}*x{axis}) "
                                                f"[{remaining} more omega candidates remain]"
                                            )

                                            # Save state BEFORE applying transform (for revert on failure)
                                            x_precond_saved_state = {
                                                "x_transform_map": dict(x_transform_map),
                                                "train_dl": datagen_train_noshuffle,
                                                "val_dl": datagen_val_noshuffle,
                                                "current_ast": current_ast,
                                                "model": model,
                                                "i": i,
                                                "current_val_loss": current_val_loss,
                                                "kind": "trig",
                                                "dual_layer_used": dual_layer_used,
                                                "feats": feats,
                                                "scale_specs": scale_specs,
                                                "parity_specs": parity_specs,
                                                "trig_spec": trig_spec,
                                                "invariance_feats": invariance_feats,
                                            }
                                            trig_precond_active = (axis, omega_to_try)

                                            # Record that we tried this omega
                                            trig_tried_omegas.setdefault(axis, []).append(omega_to_try)

                                            x_transform_map = cand_map
                                            _refresh_units_spec_for_xmap(x_transform_map)
                                            datagen_train_noshuffle = cand_train_dl
                                            datagen_val_noshuffle = cand_val_dl

                                            # Re-train from scratch but KEEP discovered structure
                                            model = None
                                            current_val_loss = None
                                            dual_layer_used = False
                                            under_protest = False
                                            # NOTE: Do NOT reset current_ast - preserve separations
                                            # (The x-transform is applied via data loaders, not AST)

                                            i = 0
                                            # Only mark trig as fully applied when all candidates exhausted
                                            if len(trig_tried_omegas.get(axis, [])) >= len(omega_candidates):
                                                x_precond_applied.add("trig")
                                            x_precond_attempts += 1
                                            did_restart = True
                                else:
                                    # All omega candidates tried, mark trig as done
                                    x_precond_applied.add("trig")

                    if did_restart:
                        if _stageA_should_restart("x-preconditioning restart"):
                            separable = True
                            continue
                        separable = False
                        break

            # -- Structural progress gate for x-preconditioning --
            # If we accepted an x-precond on val_loss but the subsequent
            # separability search made no structural changes, revert it.
            if _x_precond_structural_gate is not None:
                if not _x_precond_made_progress:
                    # No structural progress — revert x-preconditioning
                    kind = _x_precond_structural_gate.get("kind", "x-precond")
                    print(
                        f"[Stage A] {kind} x-preconditioning produced no structural changes; reverting..."
                    )
                    (x_transform_map, datagen_train_noshuffle, datagen_val_noshuffle,
                     current_ast, model, i, current_val_loss, dual_layer_used,
                     feats, scale_specs, parity_specs, trig_spec, invariance_feats
                    ) = _restore_x_precond_state(_x_precond_structural_gate)
                    _refresh_units_spec_for_xmap(x_transform_map)
                    trig_precond_active = None
                    _x_precond_structural_gate = None
                    _x_precond_made_progress = False
                    # Don't set separable=False — let the loop continue
                    # to potentially try the next x-precond type (trig after recip)
                    if _stageA_should_restart("x-preconditioning rollback"):
                        continue
                    separable = False
                    break
                else:
                    # Structural progress confirmed — keep x-preconditioning
                    print("[Stage A] x-preconditioning confirmed (structural progress made)")
                    _stageA_record_move(
                        move_kind=f"x_preconditioning_{_x_precond_structural_gate.get('kind', 'x-precond')}",
                        parent_ast=_x_precond_structural_gate.get("current_ast"),
                        candidate_ast=current_ast,
                        parent_loss=_x_precond_structural_gate.get("current_val_loss"),
                        candidate_loss=current_val_loss,
                        reason="x-preconditioning retained after structural progress",
                        risk_tags={"x_transform_active"},
                        details={"x_transform_map": dict(x_transform_map)},
                    )
                    _x_precond_structural_gate = None

            separable = False

        i += 1

    # Attach any input transforms so downstream stages can make a safe choice about weight reuse
    try:
        if model is not None:
            setattr(model, '_x_transform', x_transform_map)
        if models_multi is not None:
            for m_i in models_multi:
                setattr(m_i, "_x_transform", x_transform_map)
    except Exception:
        pass

    # Attach best_val_loss_base so caller can track global best across y-transforms.
    try:
        if model is not None and current_val_loss is not None and loss_scale > 0:
            model._best_val_loss_base = current_val_loss / loss_scale
    except Exception:
        pass

    try:
        _refresh_fit_link_original_y_certificate("Stage A final state", quiet=True)
        _attach_fit_link_certificate(model)
    except Exception:
        pass

    # Attach compact Stage-A structure signals used by y-search branch triggering.
    try:
        if model is not None:
            split_success_flag = bool(separability_success) and (
                (rest_add is not None) or (rest_mult is not None)
            )
            sep_score = float(stagea_best_sep_score)
            try:
                _idx = int(i_op)
                if 0 <= _idx < len(candidate_sep_ops) and bool(candidate_sep_ops[_idx]):
                    sep_score = max(sep_score, 0.55)
            except Exception:
                pass
            if split_success_flag:
                sep_score = max(sep_score, 0.95)
            sep_score = float(max(0.0, min(1.0, sep_score)))

            best_trig_rel_std = float("inf")
            for ts in (trig_scale_specs or []):
                try:
                    r = float(getattr(ts, "rel_std", float("inf")))
                    if math.isfinite(r):
                        best_trig_rel_std = min(best_trig_rel_std, r)
                except Exception:
                    continue
            if math.isfinite(best_trig_rel_std):
                trig_affine_conf = max(0.0, min(1.0, 1.0 - (best_trig_rel_std / 0.1)))
            else:
                trig_affine_conf = 0.0

            split_score = float(stagea_best_split_score)
            if split_success_flag:
                split_score = max(split_score, 0.95)
            split_score = float(max(0.0, min(1.0, split_score)))

            best_sep_metric = None
            try:
                if stagea_best_sep_metric is not None and math.isfinite(float(stagea_best_sep_metric)):
                    best_sep_metric = float(stagea_best_sep_metric)
            except Exception:
                best_sep_metric = None

            stagea_signals = {
                "trig_affine_conf": float(trig_affine_conf),
                "sep_score": float(sep_score),
                "best_split_score": float(split_score),
                "split_success": float(1.0 if split_success_flag else 0.0),
                "sep_candidates_seen": float(max(0, int(stagea_sep_candidates_seen))),
                "split_accept_count": float(max(0, int(stagea_split_accept_count))),
                "full_compound_compressed": float(1.0 if bool(full_compound_solved) else 0.0),
                "full_compound_solved": 0.0,
            }
            if best_sep_metric is not None:
                stagea_signals["best_sep_metric"] = float(best_sep_metric)

            # Additional structural signals used by y-search branch confirmation.
            if str(mode).lower() == "full" and bool(getattr(search_hp, "ysearch_enable", True)):
                try:
                    from .features import (
                        detect_log_hessian_quadratic,
                        detect_square_hessian_quadratic,
                        probe_output_transforms,
                    )

                    def _safe_float(v, default=0.0):
                        try:
                            f = float(v)
                            if math.isfinite(f):
                                return f
                        except Exception:
                            pass
                        return float(default)

                    _probe_dl = None
                    try:
                        _probe_dl = datagen_train_noshuffle
                    except Exception:
                        _probe_dl = None
                    if isinstance(_probe_dl, (list, tuple)):
                        _probe_dl = next((d for d in _probe_dl if d is not None), None)
                    if _probe_dl is None:
                        try:
                            _probe_dl = datagen_val_noshuffle
                        except Exception:
                            _probe_dl = None
                        if isinstance(_probe_dl, (list, tuple)):
                            _probe_dl = next((d for d in _probe_dl if d is not None), None)

                    if _probe_dl is not None:
                        th = probe_output_transforms(
                            model,
                            _probe_dl,
                            Nxvars=int(Nxvars),
                            device=device,
                            max_batches=4,
                            max_points=2048,
                        )
                        if th is not None:
                            b = getattr(th, "baseline", None)
                            if b is not None:
                                stagea_signals.update(
                                    {
                                        "simplicity_base_score": _safe_float(
                                            getattr(b, "score", 0.0), default=0.0
                                        ),
                                        "simplicity_base_domain_ok_frac": _safe_float(
                                            getattr(b, "domain_ok_frac", 0.0), default=0.0
                                        ),
                                        "simplicity_base_poly2_rms_rel": _safe_float(
                                            getattr(b, "poly2_rms_rel", float("inf")),
                                            default=float("inf"),
                                        ),
                                        "simplicity_base_rat_rms_rel": _safe_float(
                                            getattr(b, "rat_rms_rel", float("inf")),
                                            default=float("inf"),
                                        ),
                                        "simplicity_base_hess_const_rel": _safe_float(
                                            getattr(b, "hess_const_rel", float("inf")),
                                            default=float("inf"),
                                        ),
                                        "simplicity_base_scaling_rel_std": _safe_float(
                                            getattr(b, "scaling_rel_std", float("inf")),
                                            default=float("inf"),
                                        ),
                                        "simplicity_base_cross_hess_rel": _safe_float(
                                            getattr(b, "cross_hess_rel", float("inf")),
                                            default=float("inf"),
                                        ),
                                    }
                                )

                            stagea_signals["simplicity_hint_ok"] = float(
                                1.0 if bool(getattr(th, "ok", False)) else 0.0
                            )
                            stagea_signals["simplicity_score_improvement"] = _safe_float(
                                getattr(th, "score_improvement", 0.0), default=0.0
                            )
                            bspec = getattr(th, "best", None)
                            if bspec is not None:
                                stagea_signals["simplicity_best_score"] = _safe_float(
                                    getattr(bspec, "score", 0.0), default=0.0
                                )
                                stagea_signals["simplicity_best_domain_ok_frac"] = _safe_float(
                                    getattr(bspec, "domain_ok_frac", 0.0), default=0.0
                                )

                        lq = detect_log_hessian_quadratic(
                            model,
                            _probe_dl,
                            Nxvars=int(Nxvars),
                            device=device,
                            max_batches=4,
                            max_points=2048,
                        )
                        stagea_signals.update(
                            {
                                "logquad_ok": float(1.0 if bool(getattr(lq, "ok", False)) else 0.0),
                                "logquad_score": _safe_float(getattr(lq, "score", 0.0), default=0.0),
                                "logquad_rank": _safe_float(getattr(lq, "rank", 0), default=0.0),
                                "logquad_hess_const_rel": _safe_float(
                                    getattr(lq, "hess_const_rel", float("inf")),
                                    default=float("inf"),
                                ),
                                "logquad_domain_ok_frac": _safe_float(
                                    getattr(lq, "domain_ok_frac", 0.0), default=0.0
                                ),
                            }
                        )

                        sq = detect_square_hessian_quadratic(
                            model,
                            _probe_dl,
                            Nxvars=int(Nxvars),
                            device=device,
                            max_batches=4,
                            max_points=2048,
                        )
                        stagea_signals.update(
                            {
                                "squarequad_ok": float(
                                    1.0 if bool(getattr(sq, "ok", False)) else 0.0
                                ),
                                "squarequad_score": _safe_float(
                                    getattr(sq, "score", 0.0), default=0.0
                                ),
                                "squarequad_rank": _safe_float(
                                    getattr(sq, "rank", 0), default=0.0
                                ),
                                "squarequad_hess_const_rel": _safe_float(
                                    getattr(sq, "hess_const_rel", float("inf")),
                                    default=float("inf"),
                                ),
                            }
                        )
                except Exception:
                    pass

            setattr(model, "_stageA_signals", stagea_signals)
            if models_multi is not None:
                for m_i in models_multi:
                    if m_i is not None:
                        setattr(m_i, "_stageA_signals", dict(stagea_signals))
    except Exception:
        pass

    # Multi-dataset Stage-A metadata for downstream consumers (Stage B / reporting).
    try:
        if model is not None:
            per_models = (
                list(models_multi)
                if (models_multi is not None and len(models_multi) > 0)
                else [model]
            )
            per_losses = (
                list(current_val_losses)
                if (current_val_losses is not None and len(current_val_losses) == len(per_models))
                else ([float(current_val_loss)] if current_val_loss is not None else None)
            )
            setattr(model, "_stageA_models", per_models)
            setattr(model, "_stageA_val_losses", per_losses)
            setattr(model, "_stageA_val_loss_agg", float(current_val_loss) if current_val_loss is not None else None)
            setattr(model, "_stageA_agg_mode", str(agg_mode))
            setattr(model, "_stageA_agg_weights", list(agg_weights) if agg_weights is not None else None)
            setattr(model, "_stageA_dataset_ids", list(dataset_ids) if dataset_ids is not None else None)
            setattr(model, "_stageA_initial_val_loss", stageA_initial_report_val_loss)
            setattr(model, "_stageA_initial_val_losses", stageA_initial_report_val_losses)
            setattr(model, "_stageA_initial_n_params", stageA_initial_report_n_params)
            setattr(model, "_stageA_fit_link_certificate", dict(stageA_fit_link_certificate))
            setattr(
                model,
                "_stageA_original_y_certified",
                bool(stageA_fit_link_certificate.get("original_y_certified", False)),
            )
    except Exception:
        pass

    _restore_plain_random_branch_policy("Leaving Stage A without another structural move")

    # Return the current AST (already in AST form, no conversion needed)
    return _units_finalize_return(separability_success, model, rest_add, rest_mult, candidate_sep_ops, current_ast, last_resort_suggested, full_compound_solved)

__search_definitions__ = (
    "run_separability_for_transform",
)

__search_constants__ = (

)

__search_late_bindings__ = (

)
