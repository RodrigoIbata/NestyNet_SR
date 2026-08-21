# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""Terminal certificates and compound-candidate orchestration."""

from typing import TYPE_CHECKING
import copy
import math
from typing import Dict, Optional
import torch
from nestynet_sr.sr_core import ast_to_human_readable, build_monomial_ast, collect_nn_atoms
from nestynet_sr.sr_core.bridges import AtomNode, Node, clone_ast, effective_arity, eval_input_expr, get_input_exprs, has_nontrivial_input, is_trivial_input
from nestynet_sr.sr_core.carrier_units import (
    STAGEA_BUCKINGHAM_DEFERRED,
    mark_stagea_buckingham_deferred,
    stagea_provisional_unit_metadata,
)
from .ast_utils import compact_expression_repr as _compact_expression_repr
from .candidate_builders import _build_atom_input_tensor
from .features import LeafFeatures, TrigAxisSpec
from .r1_operator_certificates import R1OperatorCertificate, build_r1_certificate_replacement, r1_certificate_poly_init, scan_r1_operator_certificates
from .model_builders import build_composite_ast
from .model_selection import compute_accept_threshold as _compute_accept_threshold, resolve_acceptance_noise_floor_raw as _resolve_acceptance_noise_floor_raw
from .stagea_fit_tournament import fit_stageA_candidate_with_tournament
from .wrapper_policy import build_compound_z_variants, compound_z_wrapper_policy, should_select_compound_variant

from ._search_shadow import (
    GREEN,
    RED,
    RESET,
    YELLOW,
    _analytic_units_rejection,
    _apply_fit_link_to_model,
    _clone_reuse_leaves,
    _compound_absorbed_effective_inputs,
    _compound_buckingham_min_freedom,
    _compound_candidate_new_arity,
    _compound_candidate_preserves_separated_coordinate,
    _loss_str,
    _select_compound_z_variant_shortlist,
    _stageA_classify_iso_z_result,
    _stageA_compound_buckingham_target_dim,
    _stageA_compound_variant_shadow_only,
    _stageA_coordinate_collapse_screen,
    _stageA_record_shadow_coordinate,
    _stageA_shadow_registry,
    _stageA_shadow_unit_status,
)
from ._search_proposals import (
    _append_compound_extra_input_asts,
    _atom_compound_cols,
    _compound_ast_for_token,
    _compound_best_proposal_confidence,
    _compound_candidate_default_extra_var_idxs,
    _compound_candidate_has_confirmed_payoff,
    _compound_candidate_payoff_policy,
    _compound_extra_input_asts_after_prefactor_peel,
    _compound_overlapping_raw_extras,
    _compound_proposal_sort_key,
    _is_ast_noop_candidate,
    _is_compound_token,
    _is_passthrough_noop_candidate,
    _is_pure_1d_full_compound_ast,
    _log_compound_proposal_shortlist,
    _quick_separability_candidates,
    _retained_axis_overlap_split_confirmed,
    _stageA_append_compound_replay_proposals,
    _stageA_append_noisy_soft_monomial_compound_proposals,
    _stageA_ast_fingerprint,
    _stageA_build_compound_replay_descriptor,
    _stageA_has_meaningful_loss_improvement,
    _stageA_split_score_str,
    _stageA_split_simplicity_score,
    _stageA_schedule_gs_compound_lanes,
    _try_nontrig_for_var_quick,
    _try_stageA_composite_closure_candidate,
)
from ._search_detection import (
    _build_compound_candidate_ast,
    _detect_compound_variable_for_atom,
)
from ._search_training import (
    _build_tag_to_leaf_map,
)
from ._search_structure import (
    _stageA_append_visible_buckingham_1d_prefactor_proposals,
    _stageA_buckingham_reason_after_visible_prefactor_transaction,
    _stageA_cap_terminal_analytic_threshold,
    _stageA_composite_closure_skip_reason,
    _stageA_composite_reduces_nn_burden,
    _stageA_compound_buckingham_reason,
    _stageA_forced_monomial_reason,
    _stageA_generate_unit_prefactor_exponents,
    _stageA_normalize_nonzero_prefactor_exponents,
    _stageA_partial_forced_monomial_peel_proposal,
    _stageA_prefactor_peeled_raw_vars,
    _stageA_shadow_promotion_audit,
    _try_stageA_forced_monomial_closure_candidate,
)

if TYPE_CHECKING:
    from ._search_policy import (
        _accept_threshold_with_structural_target,
        _format_stageA_compound_shortlist_committee_report,
        _nn_split_signature,
        _stageA_compound_shortlist_committee_rank,
        _stageA_loss_budget_multiplier,
        _stageA_terminal_closure_committee_gate,
        _stageA_under_protest_threshold_cap,
    )

def _stageA_set_r1_certificate_poly(
    ast_root: Node,
    model,
    *,
    arg_tag: str,
    cert: R1OperatorCertificate,
) -> None:
    """Initialise the affine poly atom introduced by an R1 certificate."""
    try:
        from nestynet_sr.sr_search.stageB.leaf_utils import _poly_zero_and_set

        tag_to_leaf = _build_tag_to_leaf_map(ast_root, model)
        leaf = tag_to_leaf.get(str(arg_tag))
        if leaf is not None:
            _poly_zero_and_set(leaf, r1_certificate_poly_init(cert))
    except Exception:
        pass


def _try_stageA_r1_operator_certificate_candidates(
    *,
    model,
    current_ast,
    atom,
    z_expr,
    z_readable: str,
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
    units_spec=None,
    enforce_units: bool = False,
    units_reject_cb=None,
    x_train=None,
    y_teacher=None,
    y_op=None,
    y_op_inv=None,
):
    """Try cheap visible ``R^1 -> R^1`` certificates before training ``NN[z]``."""

    if x_train is None or y_teacher is None:
        return False, None, None, None
    try:
        z_vals = eval_input_expr(z_expr, x_train).reshape(-1)
        y_vals = y_teacher.reshape(-1)
    except Exception as exc:
        print(f"[Stage A R1Cert] Could not evaluate z/teacher for z={z_readable}: {exc}")
        return False, None, None, None

    rel_max = float(getattr(search_hp, "stageA_r1_operator_cert_rel_rms", 2.0e-3))
    min_domain = float(getattr(search_hp, "stageA_r1_operator_cert_min_domain_frac", 0.98))
    certs = scan_r1_operator_certificates(
        z_vals,
        y_vals,
        max_results=8,
        rel_rms_max=rel_max,
        min_domain_frac=min_domain,
        min_branch_frac=min_domain,
        min_points=128,
    )
    if not certs:
        return False, None, None, None

    skip_tag = getattr(atom, "tag", None)
    reuse_map_raw = {t: leaf for t, leaf in (tag_to_leaf or {}).items() if t != skip_tag}
    reuse_leaves = _clone_reuse_leaves(reuse_map_raw, device, dtype)

    n_params_base = int(model.num_parameters())
    max_worsening_factor = float(getattr(search_hp, "max_worsening_factor", 100.0))
    worsening_floor = float(getattr(search_hp, "worsening_floor", 1.0e-6)) * float(loss_scale)
    acceptance_noise_floor_raw = _resolve_acceptance_noise_floor_raw(lm_hp, loss_scale)

    for cert in certs:
        tag_prefix = str(getattr(atom, "tag", None) or "stageA_r1cert")
        cand_ast, arg_tag = build_r1_certificate_replacement(
            current_ast,
            atom,
            z_expr,
            cert,
            tag_prefix=tag_prefix,
        )
        if cand_ast is None or not arg_tag:
            continue
        units_reason = _analytic_units_rejection(cand_ast, units_spec, enforce_units=bool(enforce_units))
        if units_reason is not None:
            print(f"[Stage A R1Cert] Rejected {cert.label} for z={z_readable}: units: {units_reason}")
            if callable(units_reject_cb):
                units_reject_cb("stageA_r1_operator_certificate", units_reason)
            continue
        if not _stageA_composite_reduces_nn_burden(current_ast, cand_ast):
            continue

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
            _stageA_set_r1_certificate_poly(cand_ast_updated, temp_model, arg_tag=str(arg_tag), cert=cert)
        except Exception as exc:
            print(f"[Stage A R1Cert] Rejected {cert.label} for z={z_readable}: build/init failed: {exc}")
            continue

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
            label=f"R1 certificate {cert.label}",
        )

        print(
            f"[Stage A R1Cert] Trying {cert.label} before NN[z]: "
            f"z={z_readable}, {cert.transform_name}(y)≈"
            f"{cert.affine_a:.6g}*psi(z)+{cert.affine_b:.6g}, "
            f"inverse={cert.inverse_kind}, rel={cert.rel_rms:.2e}, "
            f"accept_threshold={accept_threshold:.4e}"
        )
        if structural_target:
            print(
                "[Stage A R1Cert] Structural NN simplification target enabled: "
                f"{_nn_split_signature(current_ast)} → {_nn_split_signature(cand_ast_updated)}"
            )
        if terminal_analytic_cap:
            print("[Stage A R1Cert] Terminal analytic closure: using absolute candidate cap.")
        if under_protest_cap:
            print("[Stage A R1Cert] Under-protest branch: requiring non-regressing validation loss.")

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
                f"[Stage A R1Cert] Rejected {cert.label} for z={z_readable}, "
                f"val-loss {float(best_val_loss_cand):.4e}"
            )
            continue

        passes_relative = (
            best_train_loss is None
            or best_train_loss <= 0
            or best_train_loss_cand <= max_train_degradation * best_train_loss
        )
        passes_absolute = best_train_loss_cand <= loss_target_eff
        if not passes_relative and not passes_absolute:
            degradation = best_train_loss_cand / best_train_loss if best_train_loss else float("inf")
            print(
                f"{RED}[Stage A R1Cert] Rejected{RESET} {cert.label} for z={z_readable}: "
                f"training loss {degradation:.0f}× worse than current model"
            )
            continue

        temp_opt._update_param_groups(best_param_vec)
        best_val_loss_cand = float(best_val_loss_cand)
        coe_ok, coe_reason, coe_summary = _stageA_terminal_closure_committee_gate(
            base_ast=current_ast,
            cand_ast=cand_ast_updated,
            base_model=model,
            cand_model=temp_model,
            label=f"r1_certificate:{cert.label}:{z_readable}",
            gate_kind="stageA_r1_operator_certificate",
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
                f"{RED}[Stage A R1Cert] Rejected by CoE terminal gate{RESET} "
                f"{cert.label} for z={z_readable}: {coe_reason}"
            )
            continue
        print(
            f"{GREEN}[Stage A R1Cert] Accepted{RESET} {cert.label} for z={z_readable}, "
            f"val-loss {_loss_str(best_val_loss_cand, lm_hp)}"
        )
        return True, temp_model, cand_ast_updated, best_val_loss_cand

    return False, None, None, None


def _stageA_model_reuse_by_tag(model) -> Dict[str, torch.nn.Module]:
    """Best-effort tag -> leaf reuse map for a Stage-A composite model."""
    out: Dict[str, torch.nn.Module] = {}
    try:
        leaves = list(getattr(model, "leaf", []) or [])
    except Exception:
        leaves = []
    for i, leaf in enumerate(leaves):
        if leaf is not None:
            out[f"leaf{i}"] = leaf
    return out


def _stageA_terminal_closure_rejection_reason(
    cand,
    *,
    max_inverse_trig_rational_degree: int = 1,
) -> Optional[str]:
    """Reject Stage-A terminal closures that are too expressive for fast-track.

    Stage A terminal probes are meant to catch narrow exact closures before
    another compound pass.  Low-degree rational inverse-trig closures can be
    useful evidence for reciprocal-trigonometric compound structure, but
    degree-2+ variants are flexible enough to become smooth approximants and
    should not preempt Stage B's leaf-by-leaf confirmation route.
    """
    meta = getattr(cand, "meta", None)
    if not isinstance(meta, dict):
        return None
    is_outer_rat = (
        bool(meta.get("inverse_trig_outer_rational_closure"))
        or meta.get("pattern") == "inverse_trig_outer_rational_closure"
        or meta.get("pattern_family") == "inverse_trig_outer_rational_closure"
    )
    if not is_outer_rat:
        return None
    try:
        degree = int(meta.get("rational_degree", 0))
    except Exception:
        degree = 0
    max_degree = max(0, int(max_inverse_trig_rational_degree))
    if degree > max_degree:
        return (
            "stageA-terminal-rational-degree-cap"
            f"(degree={degree}, max={max_degree})"
        )
    return None


def _try_stageA_terminal_closure_probe(
    *,
    model,
    current_ast,
    current_val_loss: float,
    datagen_train_noshuffle,
    datagen_val_noshuffle,
    device,
    dtype,
    search_hp,
    lm_hp,
    loss_target_eff: float,
    loss_scale: float = 1.0,
    model_sep_output=None,
    y_op=None,
    y_op_inv=None,
    Nxvars=None,
    dual_layer_used=None,
    x_transform_map=None,
    units_spec=None,
    enforce_units: bool = False,
):
    """Try narrow fully-analytic Stage-B closures at a Stage-A checkpoint.

    This is intentionally not a Stage-A handoff policy.  It is an opportunistic
    terminal probe: after Stage A accepts a useful coordinate compression, test
    whether the current single NN atom can already be replaced by a visible
    analytic expression.  If no no-NN candidate passes the ordinary Stage-B
    validation/units/acceptance checks, the caller continues Stage A unchanged.
    """
    try:
        nn_atoms = collect_nn_atoms(current_ast)
    except Exception:
        return False, None, None, None, ""
    if len(nn_atoms) != 1 or current_ast is not nn_atoms[0]:
        return False, None, None, None, ""

    try:
        from .stageB.engine import StageBContext, StageBState
        from .stageB.rules import (
            RuleInverseTrigOuterClosure,
            RuleInverseTrigRationalOuterClosure,
            RulePhaseContextTrigClosure,
            RulePhaseHintReciprocalTrigPower,
            RulePhaseHintTrigClosure,
        )
    except Exception as exc:
        print(f"[Stage A Terminal] Closure probe unavailable: {type(exc).__name__}: {exc}")
        return False, None, None, None, ""

    try:
        probe_epochs = min(300, max(1, int(getattr(lm_hp, "epochs", 300) or 300)))
    except Exception:
        probe_epochs = 300
    train_loader_probe = (
        datagen_train_noshuffle()
        if callable(datagen_train_noshuffle)
        else datagen_train_noshuffle
    )
    val_loader_probe = (
        datagen_val_noshuffle()
        if callable(datagen_val_noshuffle)
        else datagen_val_noshuffle
    )

    state0 = StageBState(
        root=current_ast,
        model=model,
        reuse=_stageA_model_reuse_by_tag(model),
        val_loss=float(current_val_loss),
    )
    ctx = StageBContext(
        state=state0,
        train_loader=train_loader_probe,
        val_loader=val_loader_probe,
        lm_hp=lm_hp,
        device=device,
        dtype=dtype,
        epochs_stageB=int(probe_epochs),
        loss_scale=float(loss_scale),
        loss_good_enough_raw=float(loss_target_eff),
        score_tol=0.0,
        scale_specs=[],
        scaling_by_axis={},
        trig_by_axis={},
        phase_hints=list(getattr(search_hp, "phase_hints", []) or []),
        phase_context_hints=list(getattr(search_hp, "phase_context_hints", []) or []),
        outer_link_hints=list(getattr(search_hp, "outer_link_hints", []) or []),
        verbose=True,
        fresh_nn_factory=None,
        atom_factory=None,
        disabled_patterns=set(),
        enabled_patterns=[],
        y_op_inv=y_op_inv,
        units_spec=units_spec,
        enforce_units=bool(enforce_units),
    )

    rules = [
        RuleInverseTrigRationalOuterClosure(),
        RuleInverseTrigOuterClosure(),
        RulePhaseHintTrigClosure(),
        RulePhaseHintReciprocalTrigPower(),
        RulePhaseContextTrigClosure(),
    ]

    candidates = []
    for rule in rules:
        rule_name = getattr(rule, "name", type(rule).__name__)
        try:
            targets = list(rule.iter_targets(ctx) or [])
        except Exception as exc:
            print(f"[Stage A Terminal] Rule {rule_name} target probe failed: {exc}")
            continue
        for target in targets:
            try:
                for cand in list(rule.propose(ctx, target) or []):
                    if cand is None:
                        continue
                    candidates.append((rule_name, cand))
            except Exception as exc:
                print(f"[Stage A Terminal] Rule {rule_name} proposal failed: {exc}")

    if not candidates:
        return False, None, None, None, ""

    try:
        max_outer_rat_degree = int(
            getattr(search_hp, "stageA_terminal_inverse_trig_rational_max_degree", 1)
        )
    except Exception:
        max_outer_rat_degree = 1

    print(
        f"[Stage A Terminal] Trying {len(candidates)} narrow fully-analytic "
        "closure candidate(s) before further compound exhaustion."
    )

    best = None  # (loss, label, model, root, reason)
    for rule_name, cand in candidates:
        if getattr(cand, "root", None) is None and not cand.materialise():
            continue
        terminal_reject = _stageA_terminal_closure_rejection_reason(
            cand,
            max_inverse_trig_rational_degree=max_outer_rat_degree,
        )
        if terminal_reject is not None:
            print(f"[Stage A Terminal] Skipping {cand.label}: {terminal_reject}")
            continue
        pre = ctx.precheck_candidate(rule_name, cand, record_attempt=True)
        if not pre.ok:
            print(f"[Stage A Terminal] Precheck reject ({cand.label}): {pre.reason}")
            continue
        meta = getattr(cand, "meta", {}) if isinstance(getattr(cand, "meta", None), dict) else {}
        if meta.get("log"):
            print(str(meta["log"]).replace("[Stage B", "[Stage A Terminal"))
        try:
            cand_state = ctx.fit_candidate(cand, epochs_override=int(probe_epochs))
        except Exception as exc:
            print(f"[Stage A Terminal] Candidate {cand.label} fit failed: {exc}")
            continue
        cand_loss = float(getattr(cand_state, "val_loss", float("inf")))
        print(
            f"[Stage A Terminal] Candidate {cand.label}: "
            f"val-loss={_loss_str(cand_loss, lm_hp)}"
        )
        try:
            if collect_nn_atoms(cand_state.root):
                print(f"[Stage A Terminal] Reject ({cand.label}): candidate still contains NN atoms.")
                continue
        except Exception:
            continue
        ok, reason = ctx.should_accept(cand, cand_state)
        if ok:
            ok, reason = ctx.gauge_acceptance_gate(cand, cand_state, reason)
        if not ok:
            print(f"[Stage A Terminal] Reject ({cand.label}): {reason or 'acceptance-policy'}")
            continue
        coe_ok, coe_reason, coe_summary = _stageA_terminal_closure_committee_gate(
            base_ast=current_ast,
            cand_ast=cand_state.root,
            base_model=model,
            cand_model=cand_state.model,
            label=str(cand.label),
            gate_kind="stageA_terminal_closure_probe",
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
            print(f"[Stage A Terminal] Reject ({cand.label}): {coe_reason}")
            continue
        if best is None or cand_loss < best[0]:
            best = (cand_loss, str(cand.label), cand_state.model, cand_state.root, reason or "accepted")

    if best is None:
        print("[Stage A Terminal] No fully-analytic closure passed; continuing Stage A unchanged.")
        return False, None, None, None, ""

    best_loss, best_label, best_model, best_ast, best_reason = best
    print(
        f"{GREEN}[Stage A Terminal] Accepted{RESET} {best_label}, "
        f"val-loss {_loss_str(best_loss, lm_hp)} ({best_reason})"
    )
    try:
        print(
            "[Stage A Terminal]   Current: "
            + _compact_expression_repr(best_ast, max_length=240, y_op_inv=y_op_inv)
        )
    except Exception:
        pass

    if model_sep_output is not None:
        torch.save(
            dict(
                y_op=y_op,
                y_op_inv=y_op_inv,
                Nxvars=Nxvars,
                dual_layer=dual_layer_used,
                x_transform=x_transform_map,
                model_state_dict=best_model.state_dict(),
                ast=best_ast,
                val_loss=float(best_loss),
                fit_y_link=getattr(lm_hp, "fit_y_link", None),
                fit_y_link_scale=float(getattr(lm_hp, "fit_y_link_scale", 1.0)),
            ),
            model_sep_output,
        )

    return True, best_model, best_ast, float(best_loss), best_label


def _stageA_gs_preflight_attempt_key(current_ast, proposal):
    meta = proposal[4] if len(proposal) >= 5 and isinstance(proposal[4], dict) else {}
    carrier_fp = str(
        meta.get("gs_carrier_fingerprint")
        or _stageA_ast_fingerprint(proposal[1])
    )
    return (_stageA_ast_fingerprint(current_ast), carrier_fp)


def _try_stageA_decisive_gs_preflight_for_atom(
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
    best_train_loss,
    loss_scale,
    model_sep_output,
    y_op,
    y_op_inv,
    Nxvars,
    x_transform_map,
    trig_spec,
    units_spec,
    enforce_units: bool,
    current_val_loss=None,
    stageA_under_protest: bool = False,
    units_reject_cb=None,
    scaling_features=None,
):
    """Offer one decisive GS carrier before legacy early-compound passes.

    This preflight is intentionally narrow.  It discovers only shared-bank GS
    carriers, selects at most one certified full-support coordinate using the
    normal decisive-lane policy, and trains it through the unchanged compound
    acceptance machinery.  If it is absent or fails, callers continue through
    the legacy early-scaling and early-pure-difference paths unchanged.
    """

    if (
        not isinstance(atom, AtomNode)
        or str(getattr(atom, "kind", "")).lower() != "nn"
        or int(effective_arity(atom)) <= 1
        or int(effective_arity(atom))
        > int(getattr(search_hp, "compound_max_vars", 4))
        or has_nontrivial_input(atom)
    ):
        return False, None, None, None, False, False

    gs_cfg = getattr(search_hp, "gs_config", None)
    if (
        not bool(getattr(search_hp, "enable_compound_detection", False))
        or gs_cfg is None
        or not bool(getattr(gs_cfg, "active", lambda: False)())
        or not bool(getattr(gs_cfg, "proposing", lambda: True)())
    ):
        return False, None, None, None, False, False

    leaf = tag_to_leaf.get(getattr(atom, "tag", None)) if isinstance(tag_to_leaf, dict) else None
    if leaf is None:
        return False, None, None, None, False, False

    try:
        proposals, _ = _detect_compound_variable_for_atom(
            model=model,
            atom=atom,
            leaf=leaf,
            datagen_train=datagen_train_noshuffle,
            device=device,
            max_batches=int(getattr(search_hp, "compound_max_batches", 4)),
            units_spec=units_spec,
            enforce_units=bool(enforce_units),
            gs_cfg=gs_cfg,
            gs_only=True,
        )
    except Exception as exc:
        print(f"[Stage A GS Preflight] Detection failed: {type(exc).__name__}: {exc}")
        return False, None, None, None, False, False

    try:
        gs_budget = min(
            max(0, int(getattr(gs_cfg, "max_stagea_proposals", 1))),
            max(0, int(getattr(gs_cfg, "stagea_proposal_budget", 1))),
        )
    except Exception:
        gs_budget = 1
    try:
        decisive_trials = min(
            1,
            max(0, int(getattr(gs_cfg, "decisive_stagea_max_trials", 1))),
        )
    except Exception:
        decisive_trials = 1
    decisive, _ordinary, _fallback = _stageA_schedule_gs_compound_lanes(
        proposals or [],
        max_ordinary_proposals=0,
        max_gs_proposals=gs_budget,
        decisive_min_confidence=float(
            getattr(gs_cfg, "decisive_stagea_min_confidence", 0.995)
        ),
        decisive_max_trials=decisive_trials,
    )
    if not decisive:
        return False, None, None, None, False, False

    proposal = decisive[0]
    attempt_key = _stageA_gs_preflight_attempt_key(current_ast, proposal)
    carrier_fp = str(attempt_key[1])
    attempted = getattr(search_hp, "_stageA_decisive_gs_preflight_attempted", None)
    if not isinstance(attempted, set):
        attempted = set()
        try:
            setattr(search_hp, "_stageA_decisive_gs_preflight_attempted", attempted)
        except Exception:
            attempted = set()
    if attempt_key in attempted:
        print(
            "[Stage A GS Preflight] Skipping previously attempted decisive "
            f"carrier {carrier_fp}."
        )
        return False, None, None, None, False, False
    attempted.add(attempt_key)

    try:
        z_desc = ast_to_human_readable(proposal[1], x_transform_map)
    except Exception:
        z_desc = str(proposal[1])
    print(
        "[Stage A GS Preflight] Trying one certified full-support carrier "
        f"before legacy early compounds: z={z_desc}"
    )

    return _try_compound_candidates_for_atom(
        proposals=decisive,
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
        stageA_under_protest=bool(stageA_under_protest),
        best_train_loss=best_train_loss,
        loss_scale=loss_scale,
        model_sep_output=model_sep_output,
        y_op=y_op,
        y_op_inv=y_op_inv,
        Nxvars=Nxvars,
        x_transform_map=x_transform_map,
        trig_spec=trig_spec,
        units_spec=units_spec,
        enforce_units=bool(enforce_units),
        units_reject_cb=units_reject_cb,
        allow_iterative_extension=False,
        skip_same_arity_if_already_sep=False,
        oracle_trig_specs=None,
        scaling_features=scaling_features,
        decisive_gs_only=True,
    )


def _try_stageA_compound_during_sep_for_atom(
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
    best_train_loss,
    loss_scale,
    model_sep_output,
    y_op,
    y_op_inv,
    Nxvars,
    x_transform_map,
    trig_spec,
    scale_specs,
    invariance_feats,
    trig_axis_specs_all,
    units_spec,
    enforce_units: bool,
    current_val_loss=None,
    stageA_under_protest: bool = False,
    units_reject_cb=None,
):
    """Try compound candidates inside the Stage-A separability pass.

    This is the action hook for detectors that are already informative during
    Stage A.  When a leaf has a high-confidence arity-reducing compound, train
    that candidate before accepting a separability split that would chop the
    same leaf into smaller NN representatives.
    """
    if (
        not isinstance(atom, AtomNode)
        or str(getattr(atom, "kind", "")).lower() != "nn"
        or int(effective_arity(atom)) <= 1
    ):
        return False, None, None, None, False, False

    enable_compound = bool(getattr(search_hp, "enable_compound_detection", False))
    if not enable_compound:
        return False, None, None, None, False, False

    compound_max_vars = int(getattr(search_hp, "compound_max_vars", 4))
    if not (2 <= int(effective_arity(atom)) <= compound_max_vars):
        return False, None, None, None, False, False

    leaf = tag_to_leaf.get(getattr(atom, "tag", None)) if isinstance(tag_to_leaf, dict) else None
    if leaf is None:
        return False, None, None, None, False, False

    compound_max_exponent = int(getattr(search_hp, "compound_max_exponent", 5))
    compound_threshold = float(getattr(search_hp, "compound_threshold", 0.05))
    compound_max_batches = int(getattr(search_hp, "compound_max_batches", 4))
    confidence_gate = float(getattr(search_hp, "compound_confidence_gate", 0.85))
    atom_already_compound = has_nontrivial_input(atom)
    baseline_split_score = None
    baseline_already_sep = False

    if atom_already_compound:
        try:
            inputs_cur = get_input_exprs(atom)
            z_expr_cur = inputs_cur[0]
            extra_var_idxs_cur = [
                int(inp.var_idxs[0]) for inp in inputs_cur[1:] if is_trivial_input(inp)
            ]
            extra_input_asts_cur = [inp for inp in inputs_cur[1:] if not is_trivial_input(inp)]
            if extra_var_idxs_cur or extra_input_asts_cur:
                baseline_sep_cands = _quick_separability_candidates(
                    model=model,
                    leaf=leaf,
                    z_expr=z_expr_cur,
                    extra_var_idxs=extra_var_idxs_cur,
                    extra_input_asts=extra_input_asts_cur,
                    datagen_train=datagen_train_noshuffle,
                    device=device,
                    dtype=dtype,
                )
                baseline_already_sep = bool(baseline_sep_cands)
                baseline_split_score = _stageA_split_simplicity_score(
                    sep_cands=baseline_sep_cands,
                    z_expr=z_expr_cur,
                    extra_var_idxs=extra_var_idxs_cur,
                    extra_input_asts=extra_input_asts_cur,
                    retained_axis_wrapper=False,
                    same_arity_coordinate=False,
                )
                if baseline_split_score is not None:
                    print(
                        "[Stage A Compound] Existing coordinates already expose "
                        "a split; retained-axis rewrites must be simpler "
                        f"({_stageA_split_score_str(baseline_split_score)})."
                    )
        except Exception:
            baseline_split_score = None

    try:
        proposals, oracle_trig_specs = _detect_compound_variable_for_atom(
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
    except Exception as exc:
        print(f"[Stage A Compound] In-pass detection failed: {type(exc).__name__}: {exc}")
        return False, None, None, None, False, False

    proposals = _stageA_append_compound_replay_proposals(
        proposals or [],
        search_hp=search_hp,
        lm_hp=lm_hp,
        current_ast=current_ast,
        atom=atom,
        Nxvars=Nxvars,
        x_transform_map=x_transform_map,
        units_spec=units_spec,
    )
    proposals = _stageA_append_visible_buckingham_1d_prefactor_proposals(
        proposals,
        current_ast=current_ast,
        atom=atom,
        units_spec=units_spec,
        enforce_units=bool(enforce_units),
        search_hp=search_hp,
        x_transform_map=x_transform_map,
    )
    proposals = _stageA_append_noisy_soft_monomial_compound_proposals(
        proposals,
        atom=atom,
        scaling_features=scale_specs,
        search_hp=search_hp,
        lm_hp=lm_hp,
        loss_scale=loss_scale,
    )
    if not proposals:
        return False, None, None, None, False, False
    best_conf = _compound_best_proposal_confidence(proposals)
    if best_conf < confidence_gate:
        return False, None, None, None, False, False

    print(
        f"[Stage A Compound] In-pass high-confidence proposal on "
        f"NN{list(getattr(atom, 'var_idxs', ()))} (conf={best_conf:.3f}); "
        "trying before separability split."
    )

    return _try_compound_candidates_for_atom(
        proposals=proposals,
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
        stageA_under_protest=bool(stageA_under_protest),
        best_train_loss=best_train_loss,
        loss_scale=loss_scale,
        model_sep_output=model_sep_output,
        y_op=y_op,
        y_op_inv=y_op_inv,
        Nxvars=Nxvars,
        x_transform_map=x_transform_map,
        trig_spec=trig_spec,
        units_spec=units_spec,
        enforce_units=bool(enforce_units),
        units_reject_cb=units_reject_cb,
        allow_iterative_extension=atom_already_compound,
        skip_same_arity_if_already_sep=baseline_already_sep,
        baseline_split_score=baseline_split_score,
        oracle_trig_specs=oracle_trig_specs,
        scaling_features=scale_specs,
    )


def _try_compound_candidates_for_atom(
    *,
    proposals,
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
    current_val_loss=None,
    stageA_under_protest: bool = False,
    best_train_loss=None,
    loss_scale,
    model_sep_output,
    y_op,
    y_op_inv,
    Nxvars,
    x_transform_map,
    trig_spec=None,
    units_spec=None,
    enforce_units: bool = False,
    units_reject_cb=None,
    allow_iterative_extension: bool = False,
    skip_same_arity_if_already_sep: bool = False,
    baseline_split_score=None,
    oracle_trig_specs=None,
    scaling_features=None,
    decisive_gs_only: bool = False,
):
    """Try compound-variable replacements for a single NN atom.

    Parameters
    ----------
    proposals : list[(exponents, z_ast, confidence)]
        Output of _detect_compound_variable_for_atom.
    allow_iterative_extension : bool
        If True, process the atom even if it's already compound. This is used
        for iterative structure detection (e.g., trig-wrapped compound proposals
        where z = x*y becomes w = sin(z)).
    decisive_gs_only : bool
        Try only the single GS proposal eligible for the protected decisive
        lane.  Auxiliary ordinary proposal injectors and GS fallbacks are
        suppressed so failure returns control to the untouched legacy lane.

    Returns
    -------
    accepted : bool
    new_model : torch.nn.Module | None
    new_ast : Node | None
    new_val_loss : float | None
    full_compound_solved : bool
        Back-compat flag for a pure full-variable NN[z(x)] compression.
        This is a provisional representation, not proof that the outer map f(z)
        has been identified.
    """

    proposals = list(proposals or [])
    if proposals and not bool(decisive_gs_only):
        attempted = getattr(search_hp, "_stageA_decisive_gs_preflight_attempted", None)
        if isinstance(attempted, set) and attempted:
            kept = []
            skipped = 0
            for proposal in proposals:
                meta = (
                    proposal[4]
                    if len(proposal) >= 5 and isinstance(proposal[4], dict)
                    else {}
                )
                if (
                    str(meta.get("source", "")) == "generalized_symmetry"
                    and _stageA_gs_preflight_attempt_key(current_ast, proposal) in attempted
                ):
                    skipped += 1
                    continue
                kept.append(proposal)
            proposals = kept
            if skipped:
                print(
                    "[Stage A GS Preflight] Suppressing "
                    f"{skipped} already-attempted GS carrier copy/copies in "
                    "the later compound lane."
                )
    if not bool(decisive_gs_only):
        proposals = _stageA_append_compound_replay_proposals(
            proposals,
            search_hp=search_hp,
            lm_hp=lm_hp,
            current_ast=current_ast,
            atom=atom,
            Nxvars=Nxvars,
            x_transform_map=x_transform_map,
            units_spec=units_spec,
        )
        proposals = _stageA_append_visible_buckingham_1d_prefactor_proposals(
            proposals,
            current_ast=current_ast,
            atom=atom,
            units_spec=units_spec,
            enforce_units=bool(enforce_units),
            search_hp=search_hp,
            x_transform_map=x_transform_map,
        )
        proposals = _stageA_append_noisy_soft_monomial_compound_proposals(
            proposals,
            atom=atom,
            scaling_features=scaling_features,
            search_hp=search_hp,
            lm_hp=lm_hp,
            loss_scale=loss_scale,
        )

    if not proposals:
        return False, None, None, None, False, False

    bool(getattr(search_hp, "compound_try_trig_wrappers", True))

    # Skip if already has compound input (unless allow_iterative_extension is True for iterative structure)
    if has_nontrivial_input(atom) and not allow_iterative_extension:
        return False, None, None, None, False, False

    # Parent segment count / dual-layer flags for consistency.
    parent_num_segments = atom.kwargs.get("num_segments", search_hp.num_segments_map[dual_layer_used])
    parent_dual_layer = atom.kwargs.get("dual_layer", dual_layer_used)

    # Build tag->leaf map once (robust to FreeConst atoms).
    if tag_to_leaf is None:
        tag_to_leaf = _build_tag_to_leaf_map(current_ast, model)

    original_leaf = tag_to_leaf.get(atom.tag)
    if original_leaf is None:
        print(f"[Compound] Warning: could not locate original leaf for tag={getattr(atom, 'tag', None)}")

    # Accept-threshold tightening relative to the current model.
    max_worsening_factor = float(getattr(search_hp, "max_worsening_factor", 100.0))
    worsening_floor = float(getattr(search_hp, "worsening_floor", 1.0e-6)) * loss_scale

    # --------------------------------------------------------------
    # Proposal normalization + efficiency controls
    # --------------------------------------------------------------
    max_proposals_to_try = int(getattr(search_hp, "compound_max_proposals_to_try", 6))
    max_variants_to_try = int(getattr(search_hp, "compound_max_variants_to_try", 3))

    screen_enable = bool(getattr(search_hp, "compound_variant_screen_enable", True))
    screen_bins = int(getattr(search_hp, "compound_variant_screen_bins", 64))
    screen_gate = float(getattr(search_hp, "compound_variant_screen_gate", 0.15))
    pretrain_max_points = getattr(search_hp, "compound_pretrain_max_points", None)

    # Iso-z residual dependency check (cheap sanity gate for 1D monomial compounds).
    # If the leaf truly depends only on z (or on z after peeling a monomial prefactor),
    # then varying the underlying variables along an iso-z manifold should leave the
    # leaf output invariant. This blocks "approximately 1D" but incomplete compounds.
    iso_z_enable = bool(getattr(search_hp, "compound_iso_z_enable", True))
    iso_z_threshold = float(getattr(search_hp, "compound_iso_z_threshold", 0.03))
    iso_z_quantile = float(getattr(search_hp, "compound_iso_z_quantile", 0.90))
    iso_z_n_sample = int(getattr(search_hp, "compound_iso_z_n_sample", 300))
    iso_z_n_perturb = int(getattr(search_hp, "compound_iso_z_n_perturb", 10))
    iso_z_log_t_range = float(getattr(search_hp, "compound_iso_z_log_t_range", 0.3))
    iso_z_min_valid = int(getattr(search_hp, "compound_iso_z_min_valid", 64))
    iso_z_noise_mult = float(getattr(search_hp, "compound_iso_z_noise_mult", 2.0))
    iso_z_noise_cap = float(getattr(search_hp, "compound_iso_z_noise_cap", 0.25))
    iso_z_struct_margin = float(getattr(search_hp, "compound_iso_z_struct_margin", 0.01))
    iso_z_noisy_min_conf = float(getattr(search_hp, "compound_iso_z_noisy_min_confidence", 0.75))

    # Normalize tuples to (pattern, z_ast, confidence, extra_override, meta)
    normed_proposals = []
    # Track which variables have had non-trig (z*xk) tried and whether it enabled separability.
    # Key: var_idx, Value: (tried: bool, enables_sep: bool)
    var_nontrig_tried = {}  # type: dict[int, tuple[bool, bool]]
    for p in proposals:
        if len(p) == 3:
            pattern, z_ast, conf = p
            normed_proposals.append((pattern, z_ast, float(conf), None, {"kind": "monomial"}))
        elif len(p) == 4:
            pattern, z_ast, conf, extra_override = p
            normed_proposals.append((pattern, z_ast, float(conf), extra_override, {"kind": "monomial"}))
        else:
            pattern, z_ast, conf, extra_override, meta = p
            normed_proposals.append((pattern, z_ast, float(conf), extra_override, (meta or {})))

    partial_forced_props = []
    for pattern, z_ast, conf, extra_override, meta in (
        [] if decisive_gs_only else list(normed_proposals)
    ):
        try:
            pat_t = tuple(int(v) for v in pattern)
        except Exception:
            continue
        extra_var_idxs_for_plan = (
            list(extra_override)
            if (extra_override is not None)
            else _compound_candidate_default_extra_var_idxs(atom, pat_t)
        )
        meta_extra_input_asts_for_plan = []
        try:
            for _inp in (meta or {}).get("extra_input_asts", ()) or ():
                _append_compound_extra_input_asts(
                    meta_extra_input_asts_for_plan,
                    _inp,
                    x_transform_map=x_transform_map,
                )
        except Exception:
            meta_extra_input_asts_for_plan = []
        part = _stageA_partial_forced_monomial_peel_proposal(
            current_ast=current_ast,
            atom=atom,
            z_expr=z_ast,
            pattern=pat_t,
            extra_var_idxs=extra_var_idxs_for_plan,
            extra_input_asts=meta_extra_input_asts_for_plan or None,
            confidence=float(conf),
            meta=meta if isinstance(meta, dict) else {},
            scaling_features=scaling_features,
            scaling_rel_std_threshold=float(getattr(search_hp, "oracle_scaling_rel_std", 0.08)),
            scaling_k_int_threshold=float(getattr(search_hp, "early_compound_k_int", 0.15)),
            units_spec=units_spec,
            enforce_units=bool(enforce_units),
        )
        if part is None:
            continue
        partial_forced_props.append(part)
        try:
            pmeta = part[4] if len(part) > 4 and isinstance(part[4], dict) else {}
            clean = pmeta.get("prefactor_exponents")
            full = pmeta.get("forced_full_powers")
            resid = pmeta.get("residual_local_indices")
            print(
                "[Stage A PartialPeel] Adding clean integer prefactor peel "
                f"prefactor={clean}, residual_local={resid}, full_powers={full}"
            )
        except Exception:
            pass
    normed_proposals.extend(partial_forced_props)

    scout_replay_proposals = [
        prop for prop in normed_proposals
        if isinstance(prop[4], dict) and bool(prop[4].get("coe_scout_replay", False))
    ]
    try:
        scout_replay_proposals.sort(
            key=_compound_proposal_sort_key,
            reverse=True,
        )
    except Exception:
        try:
            scout_replay_proposals.sort(
                key=lambda proposal: float(proposal[2]),
                reverse=True,
            )
        except Exception:
            pass
    local_ranked_proposals = [
        prop for prop in normed_proposals
        if not (isinstance(prop[4], dict) and bool(prop[4].get("coe_scout_replay", False)))
    ]
    gs_cfg = getattr(search_hp, "gs_config", None)
    gs_proposal_budget = max_proposals_to_try
    if gs_cfg is not None:
        try:
            gs_proposal_budget = min(
                max(0, int(getattr(gs_cfg, "max_stagea_proposals", max_proposals_to_try))),
                max(0, int(getattr(gs_cfg, "stagea_proposal_budget", max_proposals_to_try))),
            )
        except Exception:
            gs_proposal_budget = max_proposals_to_try
    decisive, ordinary_shortlist, gs_fallback = _stageA_schedule_gs_compound_lanes(
        local_ranked_proposals,
        max_ordinary_proposals=0 if decisive_gs_only else max_proposals_to_try,
        max_gs_proposals=gs_proposal_budget,
        decisive_min_confidence=float(
            getattr(gs_cfg, "decisive_stagea_min_confidence", 0.995)
            if gs_cfg is not None
            else 0.995
        ),
        decisive_max_trials=int(
            getattr(gs_cfg, "decisive_stagea_max_trials", 1)
            if gs_cfg is not None
            else 1
        ),
    )
    normed_proposals = list(decisive) + list(ordinary_shortlist)
    if decisive_gs_only and not decisive:
        return False, None, None, None, False, False
    if decisive:
        try:
            z_desc = ast_to_human_readable(decisive[0][1], x_transform_map)
        except Exception:
            z_desc = str(decisive[0][1])
        print(
            "[Stage A GS] Scheduling one decisive protected trial before the "
            f"ordinary compound lane: z={z_desc}"
        )
    if scout_replay_proposals and not decisive_gs_only:
        scout_lane_k = max(0, int(getattr(search_hp, "coe_stageA_replay_scout_lane_k", 2) or 2))
        seen_replay = set()
        for prop in normed_proposals:
            try:
                extra = prop[3] if len(prop) > 3 else None
                meta = prop[4] if len(prop) > 4 and isinstance(prop[4], dict) else {}
                seen_replay.add((
                    _stageA_ast_fingerprint(prop[1]),
                    tuple(int(v) for v in (extra or ())),
                    str(meta.get("kind", "")),
                ))
            except Exception:
                continue
        added_scout = 0
        for prop in scout_replay_proposals:
            if added_scout >= scout_lane_k:
                break
            try:
                extra = prop[3] if len(prop) > 3 else None
                meta = prop[4] if len(prop) > 4 and isinstance(prop[4], dict) else {}
                key = (
                    _stageA_ast_fingerprint(prop[1]),
                    tuple(int(v) for v in (extra or ())),
                    str(meta.get("kind", "")),
                )
            except Exception:
                key = None
            if key is not None and key in seen_replay:
                continue
            if key is not None:
                seen_replay.add(key)
            normed_proposals.append(prop)
            added_scout += 1
        if added_scout:
            print(
                f"[CoE StageA replay] Preserving {added_scout} scout proposal(s) "
                "in a separate compound shortlist lane."
            )
    if gs_fallback and not decisive_gs_only:
        normed_proposals.extend(gs_fallback)
        print(
            f"[Stage A GS] Preserving {len(gs_fallback)} fallback proposal(s) "
            "after the ordinary compound lane."
        )
    _log_compound_proposal_shortlist(
        normed_proposals,
        x_transform_map=x_transform_map,
    )

    # Build a small cache of x and teacher outputs once per atom.
    # This is used for cheap screening and for teacher-distillation pretrain.
    x_train = None
    y_teacher = None
    if original_leaf is not None and (
        screen_enable
        or int(getattr(search_hp, "compound_pretrain_epochs", 2000)) > 0
        or iso_z_enable
    ):
        try:
            xs = []
            n = 0
            max_pts = int(pretrain_max_points) if pretrain_max_points is not None else None
            for batch in datagen_train_noshuffle:
                if isinstance(batch, (list, tuple)):
                    x_batch, _ = batch
                else:
                    x_batch = batch
                if max_pts is not None:
                    remaining = max_pts - n
                    if remaining <= 0:
                        break
                    if x_batch.shape[0] > remaining:
                        x_batch = x_batch[:remaining]
                xs.append(x_batch.to(device=device, dtype=dtype))
                n += int(x_batch.shape[0])
            if xs:
                x_train = torch.cat(xs, dim=0)
                with torch.no_grad():
                    leaf_input = _build_atom_input_tensor(atom, x_train)
                    y_teacher = original_leaf(leaf_input)
        except Exception as e:
            print(f"[Compound] Could not build screening/pretrain cache: {e}")
            x_train, y_teacher = None, None

    def _screen_univariate(z_vals: torch.Tensor, y_vals: torch.Tensor, n_bins: int = 64) -> float:
        """Cheap check: does y collapse to a single-valued function of z?"""
        try:
            z = z_vals.reshape(-1)
            y = y_vals.reshape(-1)
            m = torch.isfinite(z) & torch.isfinite(y)
            z = z[m]
            y = y[m]
            N = int(z.numel())
            if N < 128:
                return 0.0
            y_mean = y.mean()
            sst = torch.sum((y - y_mean) ** 2)
            if float(sst) <= 1e-20:
                return 0.0
            idx = torch.argsort(z)
            y_sorted = y[idx]

            nb = int(max(4, min(int(n_bins), N)))
            # Equal-count bins for robustness
            step = max(1, N // nb)
            sse = torch.tensor(0.0, device=y.device, dtype=y.dtype)
            for b in range(nb):
                a = b * step
                c = N if b == nb - 1 else min(N, (b + 1) * step)
                if c <= a:
                    break
                yb = y_sorted[a:c]
                mu = yb.mean()
                sse = sse + torch.sum((yb - mu) ** 2)
            r2 = 1.0 - float((sse / (sst + 1e-30)).detach().cpu())
            if not math.isfinite(r2):
                return 0.0
            return float(max(0.0, min(1.0, r2)))
        except Exception:
            return 0.0

    def _compound_screen_coords(z_expr):
        coords = []
        try:
            coords.append(eval_input_expr(z_expr, x_train))
        except Exception:
            return []
        try:
            peeled_raw_vars = _stageA_prefactor_peeled_raw_vars(atom, prefactor_exps)
        except Exception:
            peeled_raw_vars = set()
        for idx in list(extra_var_idxs or ()):
            if int(idx) in peeled_raw_vars:
                continue
            try:
                coords.append(x_train[:, int(idx)])
            except Exception:
                pass
        for expr in list(extra_input_asts or ()):
            try:
                coords.append(eval_input_expr(expr, x_train))
            except Exception:
                pass
        return coords

    # --------------------------------------------------------------
    # Iso-z residual dependency test (1D monomial compounds)
    # --------------------------------------------------------------
    # The classic SVD confidence and the binning-based screen can be fooled when
    # the leaf is *approximately* a univariate function of z over the sampled
    # range but still depends on the raw variables (e.g. missing a leftover
    # monomial prefactor). We therefore probe invariance along iso-z manifolds:
    # vary the raw variables while holding z exactly constant and measure how
    # much the (optionally prefactor-peeled) leaf output varies.
    iso_z_cache = {}

    def _monomial_value(x_loc: torch.Tensor, exps) -> torch.Tensor:
        m = torch.ones((int(x_loc.shape[0]),), device=x_loc.device, dtype=x_loc.dtype)
        for i, e in enumerate(exps):
            try:
                ei = int(e)
            except Exception:
                continue
            if ei == 0:
                continue
            m = m * (x_loc[:, i] ** ei)
        return m

    def _iso_z_residual_ratio_monomial(pattern, prefactor_exps, y_baseline, fit_y_link, fit_y_link_scale_used):
        try:
            if (not iso_z_enable) or (original_leaf is None) or (x_train is None) or (y_baseline is None):
                return None
            pat = tuple(int(v) for v in pattern)
            k = int(len(pat))
            if k < 2:
                return 0.0

            # Participating axes in z (non-zero exponents).
            idxs = [i for i, e in enumerate(pat) if int(e) != 0]
            if len(idxs) < 2:
                return 0.0

            yb = y_baseline.reshape(-1)
            yb = yb[torch.isfinite(yb)]
            if int(yb.numel()) < max(128, int(iso_z_min_valid)):
                return None
            overall_std = torch.std(yb)
            if (not torch.isfinite(overall_std)) or float(overall_std.detach().cpu()) <= 1e-20:
                return None

            # Local inputs for this leaf.
            x_loc = _build_atom_input_tensor(atom, x_train)
            if x_loc.dim() != 2:
                return None
            if int(x_loc.shape[1]) != k:
                # Best-effort alignment (should not happen for fresh atoms).
                kk = int(min(int(x_loc.shape[1]), k))
                x_loc = x_loc[:, :kk]
                pat = pat[:kk]
                if prefactor_exps is not None:
                    prefactor_exps = tuple(prefactor_exps)[:kk]
                idxs = [i for i, e in enumerate(pat) if int(e) != 0]
                if len(idxs) < 2:
                    return 0.0
                k = kk

            # Filter to points safe for negative exponents (avoid 1/0 blowups).
            safe_eps = 1e-8
            need_nonzero = [i for i, e in enumerate(pat) if int(e) < 0]
            if prefactor_exps is not None:
                need_nonzero.extend([i for i, e in enumerate(prefactor_exps) if int(e) < 0])
            need_nonzero = sorted(set(int(i) for i in need_nonzero))
            mask = torch.isfinite(x_loc).all(dim=1)
            if need_nonzero:
                mask = mask & (torch.abs(x_loc[:, need_nonzero]) > safe_eps).all(dim=1)
            idx_all = torch.where(mask)[0]
            if int(idx_all.numel()) < int(iso_z_min_valid):
                return None

            # Bounds for each participating axis (avoid stepping out-of-domain).
            x_masked = x_loc[idx_all][:, idxs]
            x_min = torch.min(x_masked, dim=0).values
            x_max = torch.max(x_masked, dim=0).values

            # Sample a pool; we'll discard points that hit bounds under perturbation.
            pool_size = min(int(idx_all.numel()), int(max(int(iso_z_n_sample) * 4, int(iso_z_min_valid) * 2)))
            perm = idx_all[torch.randperm(int(idx_all.numel()), device=idx_all.device)[:pool_size]]
            x_pool = x_loc[perm]

            # Null-space directions in log-space: a · d = 0.
            a_cpu = torch.tensor([float(pat[i]) for i in idxs], dtype=torch.float64).view(1, -1)
            if (not torch.isfinite(a_cpu).all()) or float(torch.linalg.vector_norm(a_cpu)) <= 1e-12:
                return None
            try:
                _, _, Vh = torch.linalg.svd(a_cpu, full_matrices=True)
                dirs = Vh[1:, :].to(device=x_loc.device, dtype=x_loc.dtype)  # (n_dirs, n_vars)
            except Exception:
                return None
            if int(dirs.numel()) == 0:
                return 0.0

            K = int(max(3, int(iso_z_n_perturb)))
            tau = torch.linspace(
                -float(iso_z_log_t_range),
                +float(iso_z_log_t_range),
                K,
                device=x_loc.device,
                dtype=x_loc.dtype,
            )

            worst_ratio = 0.0
            any_dir = False
            for d in dirs:
                if float(torch.linalg.vector_norm(d).detach().cpu()) <= 1e-12:
                    continue
                scale = torch.exp(tau[:, None] * d[None, :])  # (K, n_vars)
                x_part = x_pool[:, idxs]  # (M, n_vars)
                pert = x_part[:, None, :] * scale[None, :, :]  # (M, K, n_vars)
                inb = (pert >= x_min[None, None, :]) & (pert <= x_max[None, None, :])
                valid = inb.all(dim=-1).all(dim=-1)  # (M,)
                n_valid = int(valid.sum().item())
                if n_valid < int(iso_z_min_valid):
                    continue
                any_dir = True

                valid_idx = torch.where(valid)[0]
                if int(valid_idx.numel()) > int(iso_z_n_sample):
                    valid_idx = valid_idx[torch.randperm(int(valid_idx.numel()), device=valid_idx.device)[: int(iso_z_n_sample)]]

                x_base = x_pool[valid_idx]  # (N, k)
                Np = int(x_base.shape[0])

                x_rep = x_base.unsqueeze(1).expand(-1, K, -1).clone()
                x_rep[:, :, idxs] = x_base[:, idxs].unsqueeze(1) * scale.unsqueeze(0)
                x_flat = x_rep.reshape(-1, int(x_rep.shape[-1]))

                with torch.no_grad():
                    y_hat = original_leaf(x_flat)
                if y_hat is None:
                    continue
                if y_hat.dim() == 2 and int(y_hat.shape[1]) == 1:
                    y_hat = y_hat[:, 0]
                y_hat = y_hat.reshape(-1)

                if prefactor_exps is not None:
                    try:
                        m_flat = _monomial_value(x_flat, prefactor_exps)
                        y_hat = y_hat / (m_flat + 1e-30)
                    except Exception:
                        pass

                if fit_y_link == "asinh" and fit_y_link_scale_used is not None:
                    y_hat = torch.asinh(y_hat / (float(fit_y_link_scale_used) + 1e-30))

                y_hat = y_hat.reshape(Np, K)
                std_within = torch.std(y_hat, dim=1)
                std_within = std_within[torch.isfinite(std_within)]
                if int(std_within.numel()) < max(8, int(iso_z_min_valid) // 2):
                    continue

                try:
                    qv = torch.quantile(std_within, float(iso_z_quantile))
                except Exception:
                    qv = torch.median(std_within)
                qv_f = float(qv.detach().cpu())
                ratio = qv_f / (float(overall_std.detach().cpu()) + 1e-30)
                if ratio > worst_ratio:
                    worst_ratio = ratio

            if not any_dir:
                return None
            return float(worst_ratio)
        except Exception:
            return None

    coe_compound_shortlist_enabled = bool(
        str(getattr(lm_hp, "coe_mode", "off") or "off") in {"committee_gated", "reservoir_discovery"}
    )
    coe_compound_shortlist_max = max(
        1,
        int(
            getattr(
                lm_hp,
                "coe_stageA_compound_shortlist_k",
                getattr(search_hp, "coe_stageA_compound_shortlist_k", 3),
            )
            or 3
        ),
    )
    coe_visible_prefactor_shortlist_k = max(
        0,
        int(
            getattr(
                lm_hp,
                "coe_stageA_visible_prefactor_shortlist_k",
                getattr(search_hp, "coe_stageA_visible_prefactor_shortlist_k", 1),
            )
            or 1
        ),
    )
    coe_compound_shortlist: list[dict] = []
    coe_compound_shortlist_seen: set[str] = set()

    def _coe_compound_shortlist_visible_prefactor(best_variant: dict) -> bool:
        return bool(
            best_variant.get("visible_prefactor_transaction")
            or best_variant.get("prefactor_ast_present")
            or best_variant.get("prefactor_exponents") is not None
        )

    def _coe_compound_shortlist_eligible(best_variant: dict) -> bool:
        if not coe_compound_shortlist_enabled:
            return False
        try:
            old_a = int(best_variant.get("old_arity", 0))
            new_a = int(best_variant.get("new_arity", old_a))
            if new_a >= old_a:
                return False
        except Exception:
            return False
        try:
            if not collect_nn_atoms(best_variant.get("ast")):
                return False
        except Exception:
            return False
        if bool(best_variant.get("hidden_shadow_only", False)):
            return False
        return True

    def _coe_compound_shortlist_key(best_variant: dict) -> str:
        try:
            return str(_stageA_ast_fingerprint(best_variant.get("ast")))
        except Exception:
            pass
        try:
            pat = tuple(str(v) for v in tuple(best_variant.get("pattern") or ()))
        except Exception:
            pat = ()
        return "|".join(
            [
                str(best_variant.get("z_readable", "")),
                str(best_variant.get("old_arity", "")),
                str(best_variant.get("new_arity", "")),
                repr(pat),
            ]
        )

    def _commit_compound_variant(best_variant: dict):
        best_model_compound = best_variant["model"]
        best_ast_compound = best_variant["ast"]
        best_val_loss_compound = float(best_variant["val_loss"])
        z_name_best = best_variant["z_name"]
        pattern_best = best_variant.get("pattern", None)
        kind_best = str(best_variant.get("kind", "compound"))
        z_readable_best = best_variant.get("z_readable", "")

        print(
            f"{GREEN}[Compound] Selected{RESET} best variant ({z_name_best}) ({kind_best}) "
            f"z={z_readable_best} (pattern={pattern_best}), val-loss {best_val_loss_compound:.4e}"
        )
        if kind_best == "metric_distance":
            print(
                f"{GREEN}[Stage A Metric] Selected metric NN[z] compression{RESET}: "
                f"{z_name_best}, z={z_readable_best}, val-loss {best_val_loss_compound:.4e}"
            )

        try:
            expr_str = _compact_expression_repr(best_ast_compound, max_length=240, y_op_inv=y_op_inv)
            print(f"[Stage A]   Current: {expr_str}")
        except Exception:
            pass

        coe_summary = best_variant.get("coe_stageA_compound_shortlist")
        if isinstance(coe_summary, dict):
            try:
                setattr(best_model_compound, "_stageA_coe_compound_shortlist", dict(coe_summary))
            except Exception:
                pass
        try:
            setattr(
                best_model_compound,
                "_stageA_last_compound_coe_provisional_admission",
                bool(
                    isinstance(coe_summary, dict)
                    and coe_summary.get("provisional_budget_admission", False)
                ),
            )
            setattr(
                best_model_compound,
                "_stageA_last_compound_structural_budget_multiplier",
                float(best_variant.get("structural_budget_multiplier", 1.0) or 1.0),
            )
        except Exception:
            pass
        try:
            setattr(best_model_compound, "_stageA_last_compound_old_arity", int(best_variant.get("old_arity", 0)))
        except Exception:
            pass
        try:
            setattr(best_model_compound, "_stageA_last_compound_new_arity", int(best_variant.get("new_arity", 0)))
        except Exception:
            pass
        try:
            setattr(best_model_compound, "_stageA_last_compound_kind", str(best_variant.get("kind", "")))
        except Exception:
            pass
        try:
            _pattern_for_meta = best_variant.get("pattern", None)
            if _pattern_for_meta is not None:
                setattr(best_model_compound, "_stageA_last_compound_pattern", list(_pattern_for_meta))
        except Exception:
            pass
        try:
            setattr(
                best_model_compound,
                "_stageA_last_compound_shadow_requires_payoff",
                bool(best_variant.get("shadow_requires_payoff", False)),
            )
        except Exception:
            pass
        try:
            setattr(
                best_model_compound,
                "_stageA_last_compound_shadow_visible_ast",
                bool(best_variant.get("shadow_visible_ast", False)),
            )
        except Exception:
            pass
        replay_descriptor = best_variant.get("compound_replay_descriptor")
        if isinstance(replay_descriptor, dict):
            try:
                setattr(
                    best_model_compound,
                    "_stageA_last_compound_replay_descriptor",
                    copy.deepcopy(replay_descriptor),
                )
            except Exception:
                pass
        try:
            setattr(
                best_model_compound,
                "_stageA_last_compound_was_scout_replay",
                bool(best_variant.get("coe_scout_replay", False)),
            )
        except Exception:
            pass
        carrier_unit_handoff = best_variant.get("carrier_unit_handoff")
        if isinstance(carrier_unit_handoff, dict):
            try:
                setattr(
                    best_model_compound,
                    "_stageA_last_compound_unit_handoff",
                    copy.deepcopy(carrier_unit_handoff),
                )
            except Exception:
                pass
        for attr_name, variant_key in (
            ("_stageA_last_compound_iso_z_status", "iso_z_status"),
            ("_stageA_last_compound_iso_z_ratio", "iso_z_ratio"),
            ("_stageA_last_compound_iso_z_struct_ratio", "iso_z_struct_ratio"),
            ("_stageA_last_compound_iso_z_noise_ratio", "iso_z_noise_ratio"),
            ("_stageA_last_compound_iso_z_threshold_eff", "iso_z_threshold_eff"),
            ("_stageA_last_compound_iso_z_uncertified", "iso_z_uncertified"),
            ("_stageA_last_compound_proposal_lane_protected", "proposal_lane_protected"),
        ):
            try:
                if variant_key in best_variant:
                    setattr(best_model_compound, attr_name, copy.deepcopy(best_variant.get(variant_key)))
            except Exception:
                pass

        torch.save(
            dict(
                y_op=y_op,
                y_op_inv=y_op_inv,
                Nxvars=Nxvars,
                dual_layer=dual_layer_used,
                x_transform=x_transform_map,
                model_state_dict=best_model_compound.state_dict(),
                ast=best_ast_compound,
                val_loss=best_val_loss_compound,
                fit_y_link=getattr(lm_hp, "fit_y_link", None),
                fit_y_link_scale=float(getattr(lm_hp, "fit_y_link_scale", 1.0)),
                coe_stageA_compound_shortlist=coe_summary,
                coe_stageA_compound_replay_descriptor=replay_descriptor,
                stageA_carrier_unit_handoff=carrier_unit_handoff,
            ),
            model_sep_output,
        )

        full_compound_solved = bool(
            (y_op is None)
            and _is_pure_1d_full_compound_ast(best_ast_compound, Nxvars)
        )
        best_enables_sep = bool(best_variant.get("enables_sep", False))
        return True, best_model_compound, best_ast_compound, best_val_loss_compound, full_compound_solved, best_enables_sep

    for pattern, z_ast, confidence, extra_override, meta in normed_proposals:
        meta = meta or {}
        kind = str(meta.get("kind", "compound")).lower()
        carrier_buckingham_deferred = mark_stagea_buckingham_deferred(meta)
        if (
            carrier_buckingham_deferred
            and bool(enforce_units)
            and units_spec is not None
        ):
            print(
                f"[Units/Buckingham] {STAGEA_BUCKINGHAM_DEFERRED}: "
                "certified carrier remains provisional until its outer relation is resolved."
            )

        def _provisional_carrier_unit_marker(z_expr):
            if (
                not carrier_buckingham_deferred
                or not bool(enforce_units)
                or units_spec is None
            ):
                return None
            try:
                from nestynet_sr.sr_core.units import eval_analytic_expr_dim

                return stagea_provisional_unit_metadata(
                    meta,
                    carrier_dim=eval_analytic_expr_dim(
                        z_expr,
                        units_spec.x_dims,
                    ),
                    target_dim=_stageA_compound_buckingham_target_dim(
                        current_ast,
                        atom,
                        units_spec,
                    ),
                )
            except Exception:
                return None
        is_extension_proposal = kind in ("var_times_var", "trig_times_var")
        try:
            pattern = tuple(int(v) for v in pattern)
        except Exception:
            pattern = tuple(pattern)

        # --- Just-in-time non-trig check for trig proposals ---
        # Before trying a trig proposal, first check if the simpler non-trig version
        # (z*xk instead of z*trig(xk)) would enable separability. If so, skip the trig.
        if kind == "var_times_trig":
            trig_var_idx = meta.get("trig_var_idx")
            base_z_ast = meta.get("base_z_ast")
            if trig_var_idx is not None and base_z_ast is not None:
                trig_var_idx = int(trig_var_idx)

                # Check if non-trig was already tried for this variable
                if trig_var_idx not in var_nontrig_tried:
                    # Try non-trig z*xk first (quick separability check only)
                    nontrig_enables_sep = _try_nontrig_for_var_quick(
                        base_z_ast=base_z_ast,
                        trig_var_idx=trig_var_idx,
                        extra_override=extra_override,
                        atom=atom,
                        original_leaf=original_leaf,
                        tag_to_leaf=tag_to_leaf,
                        current_ast=current_ast,
                        datagen_train_noshuffle=datagen_train_noshuffle,
                        datagen_val_noshuffle=datagen_val_noshuffle,
                        device=device,
                        dtype=dtype,
                        leaf_builder=leaf_builder,
                        parent_dual_layer=parent_dual_layer,
                        parent_num_segments=parent_num_segments,
                        search_hp=search_hp,
                        lm_hp=lm_hp,
                        loss_target_eff=loss_target_eff,
                        accept_threshold_eff_cand=accept_threshold_eff_cand,
                        best_val_loss=best_val_loss,
                        best_train_loss=best_train_loss,
                        loss_scale=loss_scale,
                        x_train=x_train,
                        y_teacher=y_teacher,
                    )
                    var_nontrig_tried[trig_var_idx] = (True, nontrig_enables_sep)

                # Skip trig if non-trig enabled separability
                _, nontrig_sep = var_nontrig_tried.get(trig_var_idx, (False, False))
                if nontrig_sep:
                    print(f"[Compound] Skipping trig for x{trig_var_idx}: z*x{trig_var_idx} enables separability")
                    continue

        elif kind == "mixed_scaling":
            # For mixed_scaling proposals: z = monomial * trig_product
            # Check if z = monomial * linear_product would enable separability instead
            trig_vars = meta.get("trig_vars", ())
            monomial_vars = meta.get("monomial_vars", ())
            monomial_exponents = meta.get("monomial_exponents", ())

            if trig_vars and monomial_vars and monomial_exponents:
                # Check each trig var - if ANY non-trig version enables sep, skip this proposal
                any_nontrig_enables_sep = False

                for trig_var_idx in trig_vars:
                    trig_var_idx = int(trig_var_idx)

                    if trig_var_idx not in var_nontrig_tried:
                        # Build base_z as just the monomial part
                        base_z_ast = build_monomial_ast(
                            tuple(monomial_vars),
                            tuple(monomial_exponents),
                        )

                        # Try non-trig z_monomial * xk
                        nontrig_enables_sep = _try_nontrig_for_var_quick(
                            base_z_ast=base_z_ast,
                            trig_var_idx=trig_var_idx,
                            extra_override=extra_override,
                            atom=atom,
                            original_leaf=original_leaf,
                            tag_to_leaf=tag_to_leaf,
                            current_ast=current_ast,
                            datagen_train_noshuffle=datagen_train_noshuffle,
                            datagen_val_noshuffle=datagen_val_noshuffle,
                            device=device,
                            dtype=dtype,
                            leaf_builder=leaf_builder,
                            parent_dual_layer=parent_dual_layer,
                            parent_num_segments=parent_num_segments,
                            search_hp=search_hp,
                            lm_hp=lm_hp,
                            loss_target_eff=loss_target_eff,
                            accept_threshold_eff_cand=accept_threshold_eff_cand,
                            best_val_loss=best_val_loss,
                            best_train_loss=best_train_loss,
                            loss_scale=loss_scale,
                            x_train=x_train,
                            y_teacher=y_teacher,
                        )
                        var_nontrig_tried[trig_var_idx] = (True, nontrig_enables_sep)

                    _, nontrig_sep = var_nontrig_tried.get(trig_var_idx, (False, False))
                    if nontrig_sep:
                        any_nontrig_enables_sep = True

                if any_nontrig_enables_sep:
                    print("[Compound] Skipping mixed_scaling trig proposal: non-trig version enables separability")
                    continue
        # --- END just-in-time non-trig check ---

        # Extra vars: either explicit override, or all vars not participating in z
        extra_var_idxs = (
            list(extra_override)
            if (extra_override is not None)
            else _compound_candidate_default_extra_var_idxs(atom, pattern)
        )

        # Extra compound-expression inputs supplied by bundle proposals.
        meta_extra_input_asts = []
        seen_extra_input_asts = set()
        for _inp in meta.get("extra_input_asts", ()) or ():
            _append_compound_extra_input_asts(
                meta_extra_input_asts,
                _inp,
                seen=seen_extra_input_asts,
                x_transform_map=x_transform_map,
            )

        # Extract preserved compound AST (extra-vs-extra PureDiff on compound atom)
        preserve_z_ast = meta.get("preserve_z_ast")
        if _compound_candidate_preserves_separated_coordinate(
            already_sep=bool(skip_same_arity_if_already_sep),
            atom=atom,
            pattern=pattern,
            preserve_z_ast=preserve_z_ast,
            extra_input_asts=meta_extra_input_asts,
        ):
            try:
                z_hr = ast_to_human_readable(z_ast, x_transform_map)
            except Exception:
                z_hr = "z"
            print(
                f"[Compound] Skipping proposal z={z_hr}: current compound already "
                "separates from its extras; preserving that separated coordinate "
                "as an NN input would compare against the wrong baseline."
            )
            continue
        # Auto-detect: if the atom already has a compound and the proposal
        # doesn't consume a compound coordinate (pattern exponent == 0),
        # preserve that coordinate.  This covers monomial/subset proposals that
        # operate on extras only; multi-input compound atoms return a token->AST
        # map, so preserve token values rather than cloning the map itself.
        if preserve_z_ast is not None:
            _append_compound_extra_input_asts(
                meta_extra_input_asts,
                preserve_z_ast,
                seen=seen_extra_input_asts,
                x_transform_map=x_transform_map,
            )
        elif has_nontrivial_input(atom):
            _cols_auto, _z_existing = _atom_compound_cols(atom)
            if _z_existing is not None:
                for _z_col, _z_tok in enumerate(_cols_auto):
                    if not _is_compound_token(_z_tok) or _z_col >= len(pattern):
                        continue
                    try:
                        _z_exp = int(pattern[_z_col])
                    except (TypeError, ValueError):
                        _z_exp = None
                    if _z_exp == 0:
                        try:
                            _append_compound_extra_input_asts(
                                meta_extra_input_asts,
                                _compound_ast_for_token(_z_existing, _z_tok),
                                seen=seen_extra_input_asts,
                                x_transform_map=x_transform_map,
                            )
                        except Exception:
                            pass
        extra_input_asts = meta_extra_input_asts or None
        prefactor_exps_raw = meta.get("prefactor_exponents", None)
        prefactor_exps = _stageA_normalize_nonzero_prefactor_exponents(
            atom,
            prefactor_exps_raw,
        )
        prefactor_ast_from_meta = meta.get("prefactor_ast", None)
        if prefactor_exps is not None or prefactor_ast_from_meta is not None:
            peeled_raw_vars = _stageA_prefactor_peeled_raw_vars(atom, prefactor_exps)
            extra_var_idxs = [
                int(v) for v in extra_var_idxs if int(v) not in peeled_raw_vars
            ]
            filtered_extra_asts = _compound_extra_input_asts_after_prefactor_peel(
                atom,
                extra_input_asts,
                prefactor_exponents=prefactor_exps,
                prefactor_ast=prefactor_ast_from_meta,
                x_transform_map=x_transform_map,
            )
            extra_input_asts = filtered_extra_asts or None

        buck_reason = _stageA_compound_buckingham_reason(
            current_ast=current_ast,
            atom=atom,
            z_expr=z_ast,
            kind=kind,
            extra_var_idxs=extra_var_idxs,
            extra_input_asts=extra_input_asts,
            units_spec=units_spec,
            enforce_units=bool(enforce_units),
            candidate_meta=meta,
        )
        partial_forced_peel = bool(meta.get("partial_forced_monomial_peel", False))
        if partial_forced_peel:
            # The bare Buckingham check is not the right object here: the
            # candidate is a visible P*NN[residual] transaction, and the full
            # AST unit check below validates that transaction.
            buck_reason = None
        generated_prefactor_transaction = False
        if buck_reason is not None and prefactor_exps is None:
            gen_pref, gen_reason = _stageA_generate_unit_prefactor_exponents(
                current_ast=current_ast,
                atom=atom,
                z_expr=z_ast,
                pattern=pattern,
                extra_var_idxs=extra_var_idxs,
                extra_input_asts=extra_input_asts,
                units_spec=units_spec,
                enforce_units=bool(enforce_units),
            )
            if gen_pref is not None:
                prefactor_exps = tuple(int(v) for v in gen_pref)
                prefactor_exps_raw = prefactor_exps
                generated_prefactor_transaction = True
                try:
                    peeled_for_log = sorted(_stageA_prefactor_peeled_raw_vars(atom, prefactor_exps))
                    residual_for_log = [
                        int(v) for v in list(extra_var_idxs or ())
                        if int(v) not in set(peeled_for_log)
                    ]
                    peeled_s = (
                        ", ".join(f"x{int(v)}" for v in peeled_for_log)
                        if peeled_for_log else "none"
                    )
                    residual_s = (
                        ", ".join(f"x{int(v)}" for v in residual_for_log)
                        if residual_for_log else "none"
                    )
                    peel_suffix = f" peeled_raw={peeled_s}; residual_raw={residual_s}"
                except Exception:
                    peel_suffix = ""
                print(
                    "[Units/Buckingham] Generated visible prefactor exponents "
                    f"{prefactor_exps} for rejected bare compound.{peel_suffix}"
                )
            elif bool(getattr(search_hp, "verbose_compound", False)):
                print(
                    "[Units/Buckingham] No generated visible prefactor complement: "
                    f"{gen_reason}"
                )

        # A generated prefactor is discovered after the initial extra-input
        # bookkeeping.  Reconcile the transaction before judging arity: a
        # coordinate exposed as P must not remain hidden inside NN[z, P].
        if prefactor_exps is not None or prefactor_ast_from_meta is not None:
            peeled_raw_vars = _stageA_prefactor_peeled_raw_vars(atom, prefactor_exps)
            extra_var_idxs = [
                int(v) for v in extra_var_idxs if int(v) not in peeled_raw_vars
            ]
            filtered_extra_asts = _compound_extra_input_asts_after_prefactor_peel(
                atom,
                extra_input_asts,
                prefactor_exponents=prefactor_exps,
                prefactor_ast=prefactor_ast_from_meta,
                x_transform_map=x_transform_map,
            )
            extra_input_asts = filtered_extra_asts or None

        overlapping_raw_extras = _compound_overlapping_raw_extras(z_ast, extra_var_idxs)
        overlap_is_retained_axis = bool(meta.get("retained_axis_wrapper", False))
        if overlapping_raw_extras and not overlap_is_retained_axis:
            try:
                z_hr = ast_to_human_readable(z_ast, x_transform_map)
            except Exception:
                z_hr = "z"
            overlap_s = ", ".join(f"x{int(v)}" for v in overlapping_raw_extras)
            print(
                f"[Compound] Skipping proposal: raw extra(s) {overlap_s} already "
                f"appear inside compound {z_hr}; overlap coordinate splits are "
                "Stage-A gauge moves, not confirmed simplifications"
            )
            continue

        # Judge structural payoff only after reconciling a visible prefactor
        # with the residual NN inputs.
        extra_var_count = int(len(extra_var_idxs))
        new_arity = _compound_candidate_new_arity(
            extra_var_count=extra_var_count,
            extra_input_asts=extra_input_asts,
        )
        old_arity = int(effective_arity(atom))
        true_1d_compound = int(new_arity) == 1
        payoff_policy = _compound_candidate_payoff_policy(old_arity, new_arity)
        if payoff_policy == "reject":
            print(
                f"[Compound] Skipping proposal: arity {old_arity} → {new_arity} "
                f"(arity increase, extras={extra_var_idxs})"
            )
            continue
        if bool(skip_same_arity_if_already_sep) and payoff_policy == "require_sep":
            print(
                f"[Compound] Skipping same-arity wrapper: arity {old_arity} → {new_arity}; "
                "current compound already separates from extras"
            )
            continue
        if payoff_policy == "require_sep":
            print(
                f"[Compound] Trying same-arity wrapper: arity {old_arity} → {new_arity}; "
                "will require confirmed separability payoff"
            )

        closure_skip_reason = _stageA_composite_closure_skip_reason(
            kind=kind,
            extra_var_idxs=extra_var_idxs,
            extra_input_asts=extra_input_asts,
            prefactor_exps=prefactor_exps_raw,
        )
        visible_prefactor_transaction = False
        if buck_reason is not None:
            buck_reason_effective = _stageA_buckingham_reason_after_visible_prefactor_transaction(
                bare_reason=buck_reason,
                current_ast=current_ast,
                atom=atom,
                z_expr=z_ast,
                pattern=pattern,
                extra_var_idxs=extra_var_idxs,
                extra_input_asts=extra_input_asts,
                prefactor_exponents=prefactor_exps,
                units_spec=units_spec,
                enforce_units=bool(enforce_units),
            )
            if buck_reason_effective is None:
                try:
                    z_readable_tx = ast_to_human_readable(z_ast, x_transform_map)
                except Exception:
                    z_readable_tx = "z"
                print(
                    "[Units/Buckingham] Bare compound rejected "
                    f"({buck_reason}); trying visible prefactor transaction P*NN[z] "
                    f"for z={z_readable_tx}"
                )
                if bool(generated_prefactor_transaction):
                    print(
                        "[Units/Buckingham] Generated prefactor transaction will be "
                        "screened by y/P collapse before LM."
                    )
                buck_reason = None
                visible_prefactor_transaction = True
            else:
                buck_reason = buck_reason_effective
        if closure_skip_reason is None:
            try:
                z_readable_closure = ast_to_human_readable(z_ast, x_transform_map)
            except Exception:
                z_readable_closure = "z"
            print(
                "[Stage A Composite] Proposal eligible for visible analytic closure: "
                f"kind={kind}, z={z_readable_closure}"
            )
            acc_closure, model_closure, ast_closure, loss_closure = (
                _try_stageA_composite_closure_candidate(
                    model=model,
                    current_ast=current_ast,
                    atom=atom,
                    z_expr=z_ast,
                    z_readable=z_readable_closure,
                    kind=kind,
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
                    buckingham_reason=buck_reason,
                )
            )
            if acc_closure:
                shadow_ok, shadow_reason = _stageA_shadow_promotion_audit(
                    base_ast=current_ast,
                    cand_ast=ast_closure,
                    old_arity=old_arity,
                    new_arity=0,
                    enables_sep=False,
                    meta=meta,
                )
                if not shadow_ok:
                    print(
                        f"[Shadow] Rejecting terminal promoted coordinate ({kind}) "
                        f"z={z_readable_closure}: {shadow_reason}"
                    )
                    continue
                if str(shadow_reason) != "not a shadow promotion":
                    print(f"[Shadow] Terminal promotion payoff confirmed: {shadow_reason}")
                return True, model_closure, ast_closure, loss_closure, False, True
        elif kind in {
            "metric_distance",
            "mixed",
            "mixed_scaling",
            "var_times_trig",
            "shadow_composite",
            "shadow_preserved_factor",
            "shadow_trig_factor_peel",
        }:
            try:
                z_readable_skip = ast_to_human_readable(z_ast, x_transform_map)
            except Exception:
                z_readable_skip = "z"
            print(
                "[Stage A Composite] Not using visible analytic closure: "
                f"kind={kind}, reason={closure_skip_reason}; z={z_readable_skip}"
            )

        if _stageA_forced_monomial_reason(buck_reason):
            acc_forced, model_forced, ast_forced, loss_forced, forced_reason = (
                _try_stageA_forced_monomial_closure_candidate(
                    model=model,
                    current_ast=current_ast,
                    atom=atom,
                    z_expr=z_ast,
                    extra_var_idxs=extra_var_idxs,
                    extra_input_asts=extra_input_asts,
                    kind=kind,
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
                    skip_power_one=(closure_skip_reason is None),
                )
            )
            if acc_forced:
                return True, model_forced, ast_forced, loss_forced, False, True
            reason_for_log = (
                f"{buck_reason}; {forced_reason}"
                if forced_reason
                else str(buck_reason)
            )
            print(f"[Units/Buckingham] Skipping compound after forced monomial probe: {reason_for_log}")
            if callable(units_reject_cb):
                units_reject_cb("compound_buckingham", reason_for_log)
            continue

        if buck_reason:
            print(f"[Units/Buckingham] Skipping compound before screening: {buck_reason}")
            if callable(units_reject_cb):
                units_reject_cb("compound_buckingham", buck_reason)
            continue

        is_compound_bundle = kind == "power_difference_bundle"
        is_full_compound_bundle = is_compound_bundle and extra_var_count == 0

        # Boost segments for true 1D compound inputs. Full simultaneous
        # difference bundles like NN[x1-x0, x3-x2] are 2D coordinate changes,
        # so keep the parent budget for now and rely on canonical init.
        if true_1d_compound:
            use_num_segments = max(
                parent_num_segments,
                int(getattr(search_hp, "compound_1d_num_segments", 32)),
            )
            print(f"[Compound] 1D compound detected (all vars in z), boosting segments to {use_num_segments}")
        elif is_full_compound_bundle:
            use_num_segments = parent_num_segments
            bundle_size = int(meta.get("bundle_size", 0) or 0)
            print(
                "[Compound] Compound bundle detected"
                f" ({bundle_size} inputs), keeping parent segments={use_num_segments}"
            )
        else:
            use_num_segments = parent_num_segments

        # Optional: if this proposal comes from log-derivative compound detection and
        # includes a monomial prefactor peel, then the *leaf target* should be teacher/prefactor.
        prefactor_ast = None
        y_teacher_for_screen = y_teacher
        if prefactor_ast_from_meta is not None:
            try:
                prefactor_ast = clone_ast(prefactor_ast_from_meta)
                if (x_train is not None) and (y_teacher is not None):
                    m_vals = eval_input_expr(prefactor_ast, x_train)
                    y_teacher_for_screen = y_teacher / (m_vals + 1e-30)
            except Exception:
                prefactor_ast = None
                y_teacher_for_screen = y_teacher
        if (
            prefactor_exps is not None
            and prefactor_ast_from_meta is None
            and (x_train is not None)
            and (y_teacher is not None)
        ):
            try:
                prefactor_exps = tuple(int(v) for v in prefactor_exps)
                if any(int(v) != 0 for v in prefactor_exps):
                    prefactor_ast = build_monomial_ast(tuple(atom.var_idxs), prefactor_exps)
                    m_vals = eval_input_expr(prefactor_ast, x_train)
                    y_teacher_for_screen = y_teacher / (m_vals + 1e-30)
            except Exception:
                prefactor_exps = meta.get("prefactor_exponents", None)
                if prefactor_ast_from_meta is None:
                    prefactor_ast = None
                y_teacher_for_screen = y_teacher

        y_teacher_for_r1_cert = y_teacher_for_screen

        # Apply fit-link transform to screening target (consistency with LM objective)
        fit_y_link = getattr(lm_hp, "fit_y_link", None)
        fit_y_link_scale_used = None
        if fit_y_link == "asinh" and y_teacher_for_screen is not None:
            fit_y_link_scale_used = getattr(lm_hp, "fit_y_link_scale", None)
            if fit_y_link_scale_used is None:
                # Compute scale from data if not set (same formula as auto-enable)
                fit_y_link_scale_used = float(torch.median(torch.abs(y_teacher_for_screen)).item()) + 1e-30
            y_teacher_for_screen = torch.asinh(y_teacher_for_screen / fit_y_link_scale_used)

        # Iso-z residual dependency check: for 1D *monomial* compounds, verify that
        # the leaf output is invariant when we vary the underlying variables along
        # iso-z manifolds (holding z exactly constant). This blocks incomplete 1D
        # compounds that only appear correct over the sampled range.
        if (
            iso_z_enable
            and true_1d_compound
            and kind == "monomial"
            and (x_train is not None)
            and (y_teacher_for_screen is not None)
            and (original_leaf is not None)
        ):
            try:
                pref_key = tuple(int(v) for v in prefactor_exps) if prefactor_exps is not None else None
            except Exception:
                pref_key = None
            try:
                scale_key = None if fit_y_link_scale_used is None else round(float(fit_y_link_scale_used), 12)
            except Exception:
                scale_key = fit_y_link_scale_used
            cache_key = (tuple(pattern), pref_key, (str(fit_y_link) if fit_y_link is not None else None), scale_key)
            ratio = iso_z_cache.get(cache_key, "__missing__")
            if ratio == "__missing__":
                ratio = _iso_z_residual_ratio_monomial(
                    pattern=pattern,
                    prefactor_exps=pref_key,
                    y_baseline=y_teacher_for_screen,
                    fit_y_link=fit_y_link,
                    fit_y_link_scale_used=fit_y_link_scale_used,
                )
                iso_z_cache[cache_key] = ratio

            if ratio is not None:
                try:
                    yb_iso = y_teacher_for_screen.reshape(-1)
                    yb_iso = yb_iso[torch.isfinite(yb_iso)]
                    y_scale_iso = float(torch.std(yb_iso).detach().cpu()) if int(yb_iso.numel()) > 1 else 0.0
                except Exception:
                    y_scale_iso = 0.0
                try:
                    noise_floor_iso = float(_resolve_acceptance_noise_floor_raw(lm_hp, loss_scale))
                except Exception:
                    noise_floor_iso = 0.0
                if not (math.isfinite(noise_floor_iso) and noise_floor_iso > 0.0):
                    try:
                        noise_floor_iso = float(getattr(lm_hp, "stageA_yspace_noise_floor_raw", 0.0) or 0.0)
                    except Exception:
                        noise_floor_iso = 0.0
                iso_z_class = _stageA_classify_iso_z_result(
                    ratio=float(ratio),
                    y_scale=float(y_scale_iso),
                    noise_floor_screen=float(noise_floor_iso),
                    clean_threshold=float(iso_z_threshold),
                    noise_mult=float(iso_z_noise_mult),
                    noise_cap=float(iso_z_noise_cap),
                    struct_margin=float(iso_z_struct_margin),
                    confidence=confidence,
                    min_confidence=float(iso_z_noisy_min_conf),
                )
                try:
                    meta.update({k: v for k, v in iso_z_class.items() if k.startswith("iso_z_")})
                    meta["iso_z_status"] = str(iso_z_class.get("status", "unknown"))
                    if bool(iso_z_class.get("iso_z_uncertified", False)):
                        meta["iso_z_uncertified"] = True
                        meta["null_verified"] = False
                        meta["provisional"] = True
                        meta["proposal_lane_protected"] = bool(
                            iso_z_class.get("proposal_lane_protected", True)
                        )
                        meta["structural_protected"] = bool(
                            iso_z_class.get("structural_protected_acceptance", True)
                        )
                except Exception:
                    pass
            if ratio is not None and float(ratio) > float(iso_z_threshold):
                try:
                    z_hr = ast_to_human_readable(z_ast)
                except Exception:
                    z_hr = str(z_ast)
                iso_status = str(meta.get("iso_z_status", "reject") or "reject")
                if iso_status == "provisional":
                    print(
                        f"[Compound] Keeping noisy-compatible 1D monomial proposal z={z_hr}: "
                        f"iso-z residual {float(ratio):.4g} > clean {float(iso_z_threshold):.4g}, "
                        f"struct={float(meta.get('iso_z_struct_ratio', float('nan'))):.4g}, "
                        f"noise={float(meta.get('iso_z_noise_ratio', float('nan'))):.4g}, "
                        f"eff={float(meta.get('iso_z_threshold_eff', float('nan'))):.4g}; "
                        "marking as provisional."
                    )
                else:
                    reason = str(meta.get("iso_z_reject_reason", "noise_adjusted_threshold_failed") or "noise_adjusted_threshold_failed")
                    if pref_key is not None:
                        print(
                            f"[Compound] Rejecting incomplete 1D monomial compound z={z_hr} "
                            f"with prefactor={pref_key}: iso-z residual {float(ratio):.4g} "
                            f"> {float(iso_z_threshold):.4g} ({reason})"
                        )
                    else:
                        print(
                            f"[Compound] Rejecting incomplete 1D monomial compound z={z_hr}: "
                            f"iso-z residual {float(ratio):.4g} > {float(iso_z_threshold):.4g} ({reason})"
                        )
                    continue

        # Leaf-level trig: derive from oracle specs (axis 0 = compound z).
        # This catches cases like f(x,y) = sin(x*y) where neither x nor y shows trig,
        # but z = x*y does.  Uses the oracle probe results already passed in.
        leaf_features = None
        if oracle_trig_specs and true_1d_compound and not is_extension_proposal:
            try:
                for ts in oracle_trig_specs:
                    if int(ts.axis) == 0:
                        leaf_features = LeafFeatures(trig_by_axis={
                            0: TrigAxisSpec(
                                axis=0, omega=ts.omega, strength=100.0,
                                n_points=ts.n_points, tmin=0.0, tmax=0.0,
                                phase=0.0 if ts.trig_fn == "cos" else math.pi / 2,
                                basis_fn=str(getattr(ts, "basis_fn", "") or getattr(ts, "trig_fn", "")),
                            )
                        })
                        print(f"[Compound] Oracle trig on z: {ts.trig_fn}, \u03c9={ts.omega:.3g}")
                        break
            except Exception as e:
                if bool(getattr(search_hp, "verbose_compound", False)):
                    print(f"[Compound] Oracle leaf feature derivation failed: {e}")
                leaf_features = None

        # For extension proposals that absorb all remaining extras into a 1D
        # compound, oracle trig specs from the parent compound are not
        # transferable — any smooth wrapper on a 1-input NN is vacuous.
        eff_oracle_trig = (
            None if (is_extension_proposal and true_1d_compound)
            else oracle_trig_specs
        )

        # Wrapper policy + variants for z (raw z plus a bounded, kind-aware wrapper set).
        # Centralised in sr_search.wrapper_policy to avoid drift with Stage-B macro logic.
        policy = compound_z_wrapper_policy(
            kind=kind,
            pattern=pattern,
            meta=meta,
            search_hp=search_hp,
            trig_spec=trig_spec,
            atom_var_idxs=list(getattr(atom, "var_idxs", ()) or ()),
            leaf_features=leaf_features,
            oracle_trig_specs=eff_oracle_trig,
        )
        z_variants = build_compound_z_variants(
            z_ast,
            kind=kind,
            pattern=pattern,
            meta=meta,
            search_hp=search_hp,
            trig_spec=trig_spec,
            atom_var_idxs=list(getattr(atom, "var_idxs", ()) or ()),
            leaf_features=leaf_features,
            oracle_trig_specs=eff_oracle_trig,
        )

        def _precheck_compound_variant_units(z_name, z_expr):
            if (not bool(enforce_units)) or (units_spec is None):
                return None

            try:
                cand_ast_probe = _build_compound_candidate_ast(
                    current_ast,
                    atom,
                    z_expr,
                    pattern,
                    extra_var_idxs_override=extra_var_idxs,
                    prefactor_exponents=prefactor_exps,
                    prefactor_ast=prefactor_ast_from_meta,
                    extra_input_asts=extra_input_asts,
                    unit_handoff_metadata=_provisional_carrier_unit_marker(
                        z_expr
                    ),
                )
            except Exception:
                return None

            try:
                from nestynet_sr.sr_core.units import check_units_ast

                ures = check_units_ast(cand_ast_probe, units_spec)
                if not bool(getattr(ures, "ok", False)):
                    return ("compound_variant", getattr(ures, "reason", "unit check failed"))
            except Exception as e:
                return ("compound_variant", f"units error: {e}")

            if bool(partial_forced_peel):
                return None
            if carrier_buckingham_deferred:
                return None

            try:
                from nestynet_sr.sr_core.units import (
                    check_compound_buckingham,
                    eval_analytic_expr_dim,
                )

                z_dim_computed = eval_analytic_expr_dim(z_expr, units_spec.x_dims)
                atom_orig_var_idxs = [int(v) for v in atom.var_idxs]
                _preserved_dims = None
                if extra_input_asts:
                    _plist = []
                    for _ea in extra_input_asts:
                        try:
                            _ed = eval_analytic_expr_dim(_ea, units_spec.x_dims)
                            if _ed is not None:
                                _plist.append(_ed)
                        except Exception:
                            pass
                    if _plist:
                        _preserved_dims = _plist
                buck_ok, buck_reason = check_compound_buckingham(
                    atom_var_idxs=atom_orig_var_idxs,
                    extra_var_idxs=extra_var_idxs,
                    z_dim=z_dim_computed,
                    x_dims=units_spec.x_dims,
                    min_freedom=_compound_buckingham_min_freedom(kind),
                    y_dim=_stageA_compound_buckingham_target_dim(current_ast, atom, units_spec),
                    extra_preserved_dims=_preserved_dims,
                )
                if not buck_ok:
                    buck_reason_effective = _stageA_buckingham_reason_after_visible_prefactor_transaction(
                        bare_reason=buck_reason,
                        current_ast=current_ast,
                        atom=atom,
                        z_expr=z_expr,
                        pattern=pattern,
                        extra_var_idxs=extra_var_idxs,
                        extra_input_asts=extra_input_asts,
                        prefactor_exponents=prefactor_exps,
                        units_spec=units_spec,
                        enforce_units=bool(enforce_units),
                    )
                    if buck_reason_effective is not None:
                        return ("compound_buckingham", buck_reason_effective)
            except Exception:
                # Keep the late Buckingham guard below as the defensive fallback.
                return None

            return None

        if bool(enforce_units) and (units_spec is not None):
            z_variants_prefiltered = []
            for nm, expr in z_variants:
                rejection = _precheck_compound_variant_units(nm, expr)
                if rejection is None:
                    z_variants_prefiltered.append((nm, expr))
                    continue
                category, reason = rejection
                try:
                    expr_desc = ast_to_human_readable(expr, x_transform_map)
                except Exception:
                    expr_desc = str(expr)
                try:
                    base_desc = ast_to_human_readable(z_ast, x_transform_map)
                except Exception:
                    base_desc = "z"
                detail = f"base_z={base_desc}, expr={expr_desc}"
                if category == "compound_buckingham":
                    print(
                        f"[Units/Buckingham] Filtering compound '{nm}' before screening: "
                        f"{detail}; {reason}"
                    )
                else:
                    print(
                        f"[Units] Filtering compound variant '{nm}' before screening: "
                        f"{detail}; {reason}"
                    )
                if callable(units_reject_cb):
                    units_reject_cb(category, reason)
            z_variants = z_variants_prefiltered

        # Optional cheap screening + cap number of variants we train.  Ordinary
        # compounds keep the historical 1D-only ranking screen; visible
        # prefactor transactions use it as a hard y/P collapse precheck.
        if (
            screen_enable
            and (x_train is not None)
            and (y_teacher_for_screen is not None)
            and (true_1d_compound or bool(visible_prefactor_transaction))
        ):
            scored = []
            for nm, expr in z_variants:
                try:
                    if bool(visible_prefactor_transaction):
                        coords = _compound_screen_coords(expr)
                        s = _stageA_coordinate_collapse_screen(coords, y_teacher_for_screen, n_bins=screen_bins)
                    else:
                        z_vals = eval_input_expr(expr, x_train)
                        s = _screen_univariate(z_vals, y_teacher_for_screen, n_bins=screen_bins)
                except Exception:
                    s = 0.0
                scored.append((nm, expr, float(s)))
            if bool(visible_prefactor_transaction):
                strict_scored = []
                for nm, expr, s in scored:
                    if float(s) >= float(screen_gate):
                        strict_scored.append((nm, expr, float(s)))
                    else:
                        try:
                            z_hr = ast_to_human_readable(expr, x_transform_map)
                        except Exception:
                            z_hr = str(nm)
                        print(
                            "[Compound] Rejecting visible-prefactor transaction before LM: "
                            f"y/P does not collapse against {nm}:{z_hr} "
                            f"(screen={float(s):.3f} < gate={float(screen_gate):.3f})"
                        )
                if not strict_scored:
                    print(
                        "[Compound] No visible-prefactor transaction variants survived "
                        "the y/P collapse screen."
                    )
                    continue
                z_variants_scored = _select_compound_z_variant_shortlist(
                    strict_scored,
                    kind=kind,
                    screen_gate=None,
                    max_variants_to_try=max_variants_to_try,
                )
            else:
                z_variants_scored = _select_compound_z_variant_shortlist(
                    scored,
                    kind=kind,
                    screen_gate=screen_gate,
                    max_variants_to_try=max_variants_to_try,
                )
        else:
            if bool(visible_prefactor_transaction):
                print(
                    "[Compound] Visible-prefactor transaction has no cheap y/P "
                    "collapse screen data; falling back to normal validation."
                )
            z_variants_scored = [(nm, expr, None) for nm, expr in z_variants]
            z_variants_scored = _select_compound_z_variant_shortlist(
                z_variants_scored,
                kind=kind,
                screen_gate=None,
                max_variants_to_try=max_variants_to_try,
            )

        best_variant = None

        # Monotonic-redundancy chains: once a lower-level member is accepted,
        # higher-level members are redundant (NN(z) = NN'(f(z)) for any
        # monotonic f, so all pure power-law wrappers are equivalent).
        _REDUNDANCY_CHAINS = (
            ("z", "rat_inv", "sq", "rat_z2", "rat_inv_z2", "rat_z4"),  # all power-law transforms of z
            ("rat_z2_over_z2m1", "rat_z4_over_z2m1_sq"),   # rational (z²-1)
            ("rat_z2_over_z2p1", "rat_z4_over_z2p1_sq"),   # rational (z²+1)
        )
        accepted_chain_levels = [None] * len(_REDUNDANCY_CHAINS)

        for z_name, z_expr, z_screen in z_variants_scored:
            # Skip redundant variants whose lower-level chain member was accepted.
            _skip = False
            for _ci, _chain in enumerate(_REDUNDANCY_CHAINS):
                if z_name in _chain:
                    _level = _chain.index(z_name)
                    if accepted_chain_levels[_ci] is not None and _level > accepted_chain_levels[_ci]:
                        print(
                            f"[Compound] Skipping {z_name} "
                            f"(redundant: {_chain[accepted_chain_levels[_ci]]} already accepted)"
                        )
                        _skip = True
                        break
            if _skip:
                continue

            # Passthrough variants can be exact no-ops on already-compound leaves.
            # Skip before any candidate build / distill / LM to avoid wasted work.
            if (
                kind == "passthrough"
                and (prefactor_exps is None)
                and (not extra_input_asts)
                and _is_passthrough_noop_candidate(
                    atom, z_expr, extra_var_idxs, x_transform_map=x_transform_map
                )
            ):
                print("[Compound] Skipping passthrough no-op candidate before training.")
                continue

            try:
                z_readable = ast_to_human_readable(z_expr, x_transform_map)
            except Exception:
                z_readable = f"{z_name}(...)"

            if _stageA_compound_variant_shadow_only(z_name):
                unit_status = _stageA_shadow_unit_status(z_expr, units_spec, bool(enforce_units))
                print(
                    f"[Compound] Shadowing wrapper variant {z_name}:{z_readable}; "
                    "wrapper-only NN coordinates are not committed in Stage A"
                )
                _stageA_record_shadow_coordinate(
                    _stageA_shadow_registry(search_hp),
                    atom=atom,
                    base_ast=z_ast,
                    shadow_ast=z_expr,
                    transform_kind=str(z_name),
                    source="compound_wrapper_variant",
                    confidence=float(confidence),
                    unit_status=unit_status,
                    evidence={
                        "compound_kind": str(kind),
                        "variant": str(z_name),
                        "old_arity": int(old_arity),
                        "new_arity": int(new_arity),
                    },
                    x_transform_map=x_transform_map,
                )
                continue

            screen_str = f", screen={float(z_screen):.3f}" if (z_screen is not None) else ""
            extras_str = f", extras={extra_var_idxs}" if extra_var_idxs else ""
            if extra_input_asts:
                extras_str += f", compound_extras={len(extra_input_asts)}"
            print(
                f"[Compound] Proposal ({kind}) conf {float(confidence):.3f}{screen_str}; "
                f"trying {z_name}:{z_readable}{extras_str}"
            )

            try:
                _has_nonzero_prefactor = (
                    prefactor_ast is not None
                    or (
                        prefactor_exps is not None
                        and any(int(v) != 0 for v in tuple(prefactor_exps))
                    )
                )
            except Exception:
                _has_nonzero_prefactor = (prefactor_exps is not None) or (prefactor_ast is not None)
            if (
                true_1d_compound
                and (not extra_var_idxs)
                and (not extra_input_asts)
                and (not _has_nonzero_prefactor)
                and (x_train is not None)
                and (y_teacher_for_r1_cert is not None)
            ):
                acc_r1, model_r1, ast_r1, loss_r1 = _try_stageA_r1_operator_certificate_candidates(
                    model=model,
                    current_ast=current_ast,
                    atom=atom,
                    z_expr=z_expr,
                    z_readable=z_readable,
                    tag_to_leaf=tag_to_leaf,
                    datagen_train_noshuffle=datagen_train_noshuffle,
                    datagen_val_noshuffle=datagen_val_noshuffle,
                    device=device,
                    dtype=dtype,
                    leaf_builder=leaf_builder,
                    parent_num_segments=use_num_segments,
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
                    units_spec=units_spec,
                    enforce_units=bool(enforce_units),
                    units_reject_cb=units_reject_cb,
                    x_train=x_train,
                    y_teacher=y_teacher_for_r1_cert,
                    y_op=y_op,
                    y_op_inv=y_op_inv,
                )
                if acc_r1:
                    return True, model_r1, ast_r1, loss_r1, False, True

            # Build candidate AST with compound input (and preserve non-participating vars as extras)
            cand_ast_compound = _build_compound_candidate_ast(
                current_ast,
                atom,
                z_expr,
                pattern,
                extra_var_idxs_override=extra_var_idxs,
                prefactor_exponents=prefactor_exps,
                prefactor_ast=prefactor_ast,
                extra_input_asts=extra_input_asts,
                unit_handoff_metadata=_provisional_carrier_unit_marker(z_expr),
            )

            if _is_ast_noop_candidate(current_ast, cand_ast_compound, x_transform_map=x_transform_map):
                print("[Compound] Skipping no-op candidate before training (AST unchanged).")
                continue

            # ----------------------------------------------------------
            # Units consistency gate (Stage A)
            #
            # Even when the leaf is an NN with unknown output units, any explicit
            # compound input expression is analytic and must be dimensionally
            # consistent (e.g., x1+x2 requires matching units; log/trig require
            # dimensionless arguments).
            # ----------------------------------------------------------
            if bool(enforce_units) and (units_spec is not None):
                try:
                    from nestynet_sr.sr_core.units import check_units_ast

                    ures = check_units_ast(cand_ast_compound, units_spec)
                    if not bool(getattr(ures, "ok", False)):
                        reason = getattr(ures, "reason", "unit check failed")
                        print(f"[Units] Skipping compound variant '{z_name}' due to units: {reason}")
                        if callable(units_reject_cb):
                            units_reject_cb("compound_variant", reason)
                        continue
                except Exception as e:
                    print(f"[Units] Skipping compound variant '{z_name}' due to units error: {e}")
                    if callable(units_reject_cb):
                        units_reject_cb("compound_variant", e)
                    continue

            # ----------------------------------------------------------
            # Buckingham π freedom check: reject compounds that
            # destroy all dimensionless degrees of freedom,
            # forcing the child into a pure monomial.
            # ----------------------------------------------------------
            if (
                bool(enforce_units)
                and (units_spec is not None)
                and (not bool(partial_forced_peel))
                and (not carrier_buckingham_deferred)
            ):
                try:
                    from nestynet_sr.sr_core.units import (
                        check_compound_buckingham,
                        eval_analytic_expr_dim,
                    )
                    z_dim_computed = eval_analytic_expr_dim(z_expr, units_spec.x_dims)
                    atom_orig_var_idxs = [int(v) for v in atom.var_idxs]
                    # Compute dims of preserved compound ASTs (existing
                    # compounds kept alongside the new one).
                    _preserved_dims = None
                    if extra_input_asts:
                        _plist = []
                        for _ea in extra_input_asts:
                            try:
                                _ed = eval_analytic_expr_dim(_ea, units_spec.x_dims)
                                if _ed is not None:
                                    _plist.append(_ed)
                            except Exception:
                                pass
                        if _plist:
                            _preserved_dims = _plist
                    buck_ok, buck_reason = check_compound_buckingham(
                        atom_var_idxs=atom_orig_var_idxs,
                        extra_var_idxs=extra_var_idxs,
                        z_dim=z_dim_computed,
                        x_dims=units_spec.x_dims,
                        min_freedom=_compound_buckingham_min_freedom(kind),
                        y_dim=_stageA_compound_buckingham_target_dim(current_ast, atom, units_spec),
                        extra_preserved_dims=_preserved_dims,
                    )
                    if not buck_ok:
                        buck_reason_effective = _stageA_buckingham_reason_after_visible_prefactor_transaction(
                            bare_reason=buck_reason,
                            current_ast=current_ast,
                            atom=atom,
                            z_expr=z_expr,
                            pattern=pattern,
                            extra_var_idxs=extra_var_idxs,
                            extra_input_asts=extra_input_asts,
                            prefactor_exponents=prefactor_exps,
                            units_spec=units_spec,
                            enforce_units=bool(enforce_units),
                        )
                        if buck_reason_effective is not None:
                            print(f"[Units/Buckingham] Skipping compound '{z_name}': {buck_reason_effective}")
                            if callable(units_reject_cb):
                                units_reject_cb("compound_buckingham", buck_reason_effective)
                            continue
                except Exception as e:
                    print(f"[Units/Buckingham] Warning: check failed for '{z_name}': {e}")
                    # Don't block on check failures — fall through

            # Reuse all existing leaves except the one being modified.
            skip_tag = getattr(atom, "tag", None)
            reuse_map_raw = {t: leaf for t, leaf in (tag_to_leaf or {}).items() if t != skip_tag}
            reuse_leaves = _clone_reuse_leaves(reuse_map_raw, device, dtype)

            try:
                temp_model_compound, _, cand_ast_compound_updated = build_composite_ast(
                    cand_ast_compound,
                    use_num_segments,
                    dual_layer=parent_dual_layer,
                    leaf_builder=leaf_builder,
                    device=device,
                    dtype=dtype,
                    reuse_leaves=reuse_leaves,
                )
                temp_model_compound = _apply_fit_link_to_model(temp_model_compound, lm_hp)

                # Teacher distillation init: original leaf -> compound leaf
                try:
                    tag_to_leaf_compound = _build_tag_to_leaf_map(cand_ast_compound_updated, temp_model_compound)
                    compound_leaf = tag_to_leaf_compound.get(atom.tag)
                except Exception:
                    compound_leaf = None

                if (original_leaf is not None) and (compound_leaf is not None) and (x_train is not None):
                    try:
                        from .training import pretrain_compound_leaf_from_teacher

                        # Use the cached x_train built once per atom.
                        temp_model_compound = pretrain_compound_leaf_from_teacher(
                            compound_model=temp_model_compound,
                            original_leaf=original_leaf,
                            compound_leaf=compound_leaf,
                            z_ast=z_expr,
                            x_data=x_train,
                            original_var_idxs=list(atom.var_idxs),
                            device=device,
                            dtype=dtype,
                            extra_var_idxs=extra_var_idxs,
                            extra_input_asts=extra_input_asts,
                            prefactor_ast=prefactor_ast,
                            original_input_asts=get_input_exprs(atom),
                            epochs=int(getattr(search_hp, "compound_pretrain_epochs", 2000)),
                            verbose=bool(getattr(search_hp, "compound_pretrain_verbose", True)),
                        )
                    except Exception as e:
                        print(f"[Compound] Pretrain from teacher failed: {e}")

                n_params_base = int(model.num_parameters())
                n_params_cand = int(temp_model_compound.num_parameters())

                # For extension, scale worsening cap with number of vars absorbed
                effective_worsening = max_worsening_factor
                if allow_iterative_extension:
                    n_absorbed = _compound_absorbed_effective_inputs(atom, new_arity)
                    if n_absorbed > 0:
                        effective_worsening = 10.0 ** (0.5 * min(n_absorbed, 3) + 0.5)

                acceptance_noise_floor_raw = _resolve_acceptance_noise_floor_raw(lm_hp, loss_scale)
                accept_threshold_compound = _compute_accept_threshold(
                    base_loss=best_val_loss,
                    best_loss=best_val_loss,
                    base_ast=current_ast,
                    cand_ast=cand_ast_compound_updated,
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
                    # Fresh compound detection: separability-like (floor semantics).
                    # Extension: use cap semantics so structural budget determines
                    # headroom, capped by absorption-scaled factor.
                    is_separability=not allow_iterative_extension,
                    max_worsening_factor=effective_worsening,
                    worsening_floor=worsening_floor,
                    noise_floor=float(acceptance_noise_floor_raw),
                )
                accept_threshold_compound, structural_target_compound = (
                    _accept_threshold_with_structural_target(
                        base_ast=current_ast,
                        cand_ast=cand_ast_compound_updated,
                        accept_threshold=accept_threshold_compound,
                        loss_target_eff=loss_target_eff,
                    )
                )
                accept_threshold_compound, under_protest_cap_compound = _stageA_under_protest_threshold_cap(
                    accept_threshold=accept_threshold_compound,
                    current_val_loss=current_val_loss if current_val_loss is not None else best_val_loss,
                    loss_floor=loss_target_eff,
                    noise_floor=acceptance_noise_floor_raw,
                    under_protest=bool(stageA_under_protest),
                    label=f"compound {z_name}",
                )

                print(
                    f"[Compound] Training candidate ({z_name}); accept_threshold={accept_threshold_compound:.4e}"
                )
                if structural_target_compound:
                    print(
                        "[Compound] Structural arity reduction target enabled: "
                        f"arity signature {_nn_split_signature(current_ast)}"
                        f" → {_nn_split_signature(cand_ast_compound_updated)}, "
                        f"target-quality threshold {accept_threshold_compound:.4e}"
                    )
                if under_protest_cap_compound:
                    print("[Compound] Under-protest branch: requiring non-regressing validation loss.")

                max_train_degradation = float(getattr(search_hp, "max_train_degradation", 100.0))
                lane_train_loss_cap = (
                    float("inf")
                    if best_train_loss is None or best_train_loss <= 0
                    else max(max_train_degradation * best_train_loss, loss_target_eff)
                )

                accepted_compound, best_val_loss_compound, best_train_loss_compound, best_param_vec_compound, temp_opt_compound = fit_stageA_candidate_with_tournament(
                    temp_model_compound,
                    datagen_train_noshuffle,
                    datagen_val_noshuffle,
                    epochs=lm_hp.epochs,
                    LM_strategy=lm_hp.strategy,
                    nval_patience=lm_hp.nval_patience,
                    loss_target=loss_target_eff,
                    accept_threshold=accept_threshold_compound,
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

                if accepted_compound:
                    # Training loss sanity check (same as Early Compound and separability)
                    passes_relative = (
                        best_train_loss is None
                        or best_train_loss <= 0
                        or best_train_loss_compound <= max_train_degradation * best_train_loss
                    )
                    passes_absolute = best_train_loss_compound <= loss_target_eff

                    if not passes_relative and not passes_absolute:
                        degradation = best_train_loss_compound / best_train_loss if best_train_loss else float('inf')
                        print(
                            f"{RED}[Compound] Rejected{RESET} ({z_name}): training loss {degradation:.0f}× worse than current model"
                        )
                        continue  # Skip to next z_variant

                    temp_opt_compound._update_param_groups(best_param_vec_compound)
                    best_val_loss_compound = float(best_val_loss_compound)

                    print(
                        f"{GREEN}[Compound] LM fit passed{RESET} ({z_name}) ({kind}) z={z_readable}, val-loss {_loss_str(best_val_loss_compound, lm_hp)}"
                    )
                    if kind == "metric_distance":
                        print(
                            f"{GREEN}[Stage A Metric] LM fit passed NN[z] compression{RESET} "
                            f"({z_name}) z={z_readable}, val-loss {_loss_str(best_val_loss_compound, lm_hp)}"
                        )

                    # Check if this wrapper enables separability (z separates from extras)
                    sep_cands_quick = _quick_separability_candidates(
                        model=temp_model_compound,
                        leaf=compound_leaf,
                        z_expr=z_expr,
                        extra_var_idxs=extra_var_idxs,
                        extra_input_asts=extra_input_asts,
                        datagen_train=datagen_train_noshuffle,
                        device=device,
                        dtype=dtype,
                    )
                    enables_sep = bool(sep_cands_quick)
                    if enables_sep:
                        print(f"[Compound] Wrapper {z_name} enables separability (z separates from extras)")

                    if overlapping_raw_extras and bool(meta.get("retained_axis_wrapper", False)):
                        retained_axis = meta.get("retained_axis", None)
                        if retained_axis is None:
                            print(
                                f"[Compound] Rejected ({z_name}) ({kind}) z={z_readable}: "
                                "retained-axis overlap has no retained axis metadata"
                            )
                            continue
                        confirmed_overlap, overlap_reason = _retained_axis_overlap_split_confirmed(
                            sep_cands=sep_cands_quick,
                            leaf=compound_leaf,
                            z_expr=z_expr,
                            extra_var_idxs=extra_var_idxs,
                            extra_input_asts=extra_input_asts,
                            retained_axis=int(retained_axis),
                            datagen_train=datagen_train_noshuffle,
                            device=device,
                            dtype=dtype,
                            search_hp=search_hp,
                        )
                        if not confirmed_overlap:
                            print(
                                f"[Compound] Rejected ({z_name}) ({kind}) z={z_readable}: "
                                f"retained-axis overlap not confirmed ({overlap_reason})"
                            )
                            continue
                        if str(overlap_reason).startswith("retained-axis effectively constant"):
                            # A k≈0 retained raw axis is evidence that the
                            # axis should disappear, not a genuine
                            # multiplicative factor.  Keep the wrapper only
                            # as a possible arity-reducing stepping stone;
                            # do not let it satisfy same-arity payoff gates.
                            enables_sep = False
                        print(
                            f"[Compound] Retained-axis overlap confirmed for x{int(retained_axis)}: "
                            f"{overlap_reason}"
                        )
                        if baseline_split_score is not None and payoff_policy == "require_sep":
                            candidate_split_score = _stageA_split_simplicity_score(
                                sep_cands=sep_cands_quick,
                                z_expr=z_expr,
                                extra_var_idxs=extra_var_idxs,
                                extra_input_asts=extra_input_asts,
                                retained_axis_wrapper=True,
                                same_arity_coordinate=True,
                            )
                            loss_reference_for_simplicity = (
                                current_val_loss if current_val_loss is not None else best_val_loss
                            )
                            meaningful_loss_win = _stageA_has_meaningful_loss_improvement(
                                cand_loss=best_val_loss_compound,
                                reference_loss=loss_reference_for_simplicity,
                                loss_floor=loss_target_eff,
                                noise_floor=acceptance_noise_floor_raw,
                            )
                            if (
                                not meaningful_loss_win
                                and (
                                    candidate_split_score is None
                                    or candidate_split_score >= baseline_split_score
                                )
                            ):
                                print(
                                    f"[Compound] Rejected ({z_name}) ({kind}) z={z_readable}: "
                                    "existing visible split is simpler than retained-axis rewrite "
                                    f"(baseline {_stageA_split_score_str(baseline_split_score)}; "
                                    f"candidate {_stageA_split_score_str(candidate_split_score)})"
                                )
                                continue
                            if (
                                meaningful_loss_win
                                and candidate_split_score is not None
                                and candidate_split_score >= baseline_split_score
                            ):
                                try:
                                    ref_s = f"{float(loss_reference_for_simplicity):.4e}"
                                except Exception:
                                    ref_s = str(loss_reference_for_simplicity)
                                print(
                                    f"[Compound] Retained-axis rewrite is structurally less simple "
                                    "than the existing visible split, but kept because validation loss "
                                    f"improves meaningfully ({ref_s} -> {best_val_loss_compound:.4e})"
                                )

                    shadow_ok, shadow_reason = _stageA_shadow_promotion_audit(
                        base_ast=current_ast,
                        cand_ast=cand_ast_compound_updated,
                        old_arity=old_arity,
                        new_arity=new_arity,
                        enables_sep=bool(enables_sep),
                        meta=meta,
                    )
                    if not shadow_ok:
                        print(
                            f"[Shadow] Rejecting promoted coordinate ({z_name}) z={z_readable}: "
                            f"{shadow_reason}"
                        )
                        continue
                    if str(shadow_reason) != "not a shadow promotion":
                        print(f"[Shadow] Promotion payoff confirmed: {shadow_reason}")

                    if not _compound_candidate_has_confirmed_payoff(
                        old_arity=old_arity,
                        new_arity=new_arity,
                        enables_sep=bool(enables_sep),
                    ):
                        print(
                            f"[Compound] Rejected ({z_name}) ({kind}) z={z_readable}: "
                            "same-arity wrapper has no confirmed separability payoff"
                        )
                        continue

                    # Track accepted chain levels only after all structural /
                    # gauge confirmation checks have passed.  A fitted wrapper
                    # that is later rejected must not suppress aliases such as
                    # rat_inv or rat_z2.
                    for _ci, _chain in enumerate(_REDUNDANCY_CHAINS):
                        if z_name in _chain:
                            _level = _chain.index(z_name)
                            if accepted_chain_levels[_ci] is None or _level < accepted_chain_levels[_ci]:
                                accepted_chain_levels[_ci] = _level

                    # Update var_nontrig_tried when var_times_var proposals are processed
                    if kind == "var_times_var":
                        absorbed_var = meta.get("absorbed_var_idx")
                        if absorbed_var is not None:
                            var_nontrig_tried[int(absorbed_var)] = (True, bool(enables_sep))

                    # Track best accepted variant for this exponent pattern.
                    # Selection policy is centralised in sr_search.wrapper_policy.
                    if should_select_compound_variant(
                        best_variant,
                        z_name=str(z_name),
                        val_loss=float(best_val_loss_compound),
                        enables_sep=bool(enables_sep),
                        policy=policy,
                    ):
                        replay_descriptor = _stageA_build_compound_replay_descriptor(
                            current_ast=current_ast,
                            atom=atom,
                            pattern=pattern,
                            z_expr=z_expr,
                            extra_var_idxs=extra_var_idxs,
                            extra_input_asts=extra_input_asts,
                            meta=meta,
                            old_arity=int(old_arity),
                            new_arity=int(new_arity),
                            confidence=float(confidence),
                            z_name=str(z_name),
                            search_hp=search_hp,
                            lm_hp=lm_hp,
                            Nxvars=Nxvars,
                            x_transform_map=x_transform_map,
                            units_spec=units_spec,
                            prefactor_exps=prefactor_exps,
                            prefactor_ast=prefactor_ast,
                        )
                        best_variant = {
                            "z_name": z_name,
                            "model": temp_model_compound,
                            "ast": cand_ast_compound_updated,
                            "val_loss": best_val_loss_compound,
                            "pattern": pattern,
                            "kind": kind,
                            "enables_sep": enables_sep,
                            "z_readable": z_readable,
                            "old_arity": int(old_arity),
                            "new_arity": int(new_arity),
                            "structural_accept_threshold": float(
                                accept_threshold_compound
                            ),
                            "reference_base_loss": float(best_val_loss),
                            "structural_budget_multiplier": _stageA_loss_budget_multiplier(
                                base_loss=best_val_loss,
                                allowed_loss=accept_threshold_compound,
                                noise_floor=acceptance_noise_floor_raw,
                            ),
                            "confidence": float(confidence),
                            "screen": None if z_screen is None else float(z_screen),
                            "hidden_shadow_only": bool(meta.get("hidden_shadow_only", False)),
                            "iso_z_status": meta.get("iso_z_status"),
                            "iso_z_ratio": meta.get("iso_z_ratio"),
                            "iso_z_struct_ratio": meta.get("iso_z_struct_ratio"),
                            "iso_z_noise_ratio": meta.get("iso_z_noise_ratio"),
                            "iso_z_threshold_eff": meta.get("iso_z_threshold_eff"),
                            "iso_z_uncertified": bool(meta.get("iso_z_uncertified", False)),
                            "proposal_lane_protected": bool(meta.get("proposal_lane_protected", False)),
                            "structural_protected": bool(meta.get("structural_protected", False)),
                            "compound_replay_descriptor": replay_descriptor,
                            "coe_scout_replay": bool(meta.get("coe_scout_replay", False)),
                            "carrier_unit_handoff": _provisional_carrier_unit_marker(
                                z_expr
                            ),
                            "visible_prefactor_transaction": bool(
                                prefactor_exps is not None or prefactor_ast is not None
                            ),
                            "prefactor_exponents": (
                                None
                                if prefactor_exps is None
                                else tuple(str(v) for v in tuple(prefactor_exps))
                            ),
                            "prefactor_ast_present": bool(prefactor_ast is not None),
                            "prefactor_pi_gauge_abs": int(meta.get("prefactor_pi_gauge_abs", 0) or 0),
                            "prefactor_pi_gauge_shift": int(meta.get("prefactor_pi_gauge_shift", 0) or 0),
                            "prefactor_pi_gauge_canonical_exponents": tuple(
                                meta.get("prefactor_pi_gauge_canonical_exponents", ()) or ()
                            ),
                        }
                else:
                    print(
                        f"[Compound] Rejected ({z_name}) ({kind}) z={z_readable}, val-loss {float(best_val_loss_compound):.4e}"
                    )
                    if kind == "metric_distance":
                        print(
                            f"[Stage A Metric] Rejected NN[z] compression "
                            f"({z_name}) z={z_readable}, val-loss {float(best_val_loss_compound):.4e}"
                        )
            except Exception as e:
                if bool(getattr(search_hp, "verbose_compound", False)):
                    print(f"[Compound] Variant '{z_name}' failed: {e}")
                continue

        if best_variant is None:
            continue

        # Reject no-op variants that preserve the exact same AST structure.
        # This avoids Stage A restart loops when an "accepted" candidate is a
        # passthrough of the current structure.
        if _is_ast_noop_candidate(current_ast, best_variant["ast"]):
            print("[Compound] Selected variant leaves AST unchanged; skipping no-op candidate.")
            continue

        if _coe_compound_shortlist_eligible(best_variant):
            shortlist_key = _coe_compound_shortlist_key(best_variant)
            if shortlist_key in coe_compound_shortlist_seen:
                print(
                    "[CoE StageA compound] Skipping duplicate strict arity-reducing "
                    f"candidate {best_variant.get('z_name')} ({best_variant.get('kind')}) "
                    f"z={best_variant.get('z_readable', '')}"
                )
                continue
            coe_compound_shortlist_seen.add(shortlist_key)
            coe_compound_shortlist.append(dict(best_variant))
            visible_prefactor_count = sum(
                1 for row in coe_compound_shortlist if _coe_compound_shortlist_visible_prefactor(row)
            )
            print(
                "[CoE StageA compound] Shortlisted strict arity-reducing candidate "
                f"{best_variant.get('z_name')} ({best_variant.get('kind')}) "
                f"arity {best_variant.get('old_arity')}->{best_variant.get('new_arity')}, "
                f"val-loss {float(best_variant.get('val_loss')):.4e} "
                f"({len(coe_compound_shortlist)}/{coe_compound_shortlist_max})"
            )
            visible_lane_satisfied = bool(
                coe_visible_prefactor_shortlist_k <= 0
                or visible_prefactor_count >= coe_visible_prefactor_shortlist_k
            )
            hard_shortlist_cap = coe_compound_shortlist_max + max(0, coe_visible_prefactor_shortlist_k)
            if len(coe_compound_shortlist) >= coe_compound_shortlist_max and (
                visible_lane_satisfied or len(coe_compound_shortlist) >= hard_shortlist_cap
            ):
                print(
                    "[CoE StageA compound] Shortlist budget reached; ranking collected candidates."
                )
                break
            continue

        if coe_compound_shortlist_enabled and coe_compound_shortlist:
            print(
                "[CoE StageA compound] Ranking strict arity-reducing shortlist "
                "before considering a non-shortlisted compound candidate."
            )
            break

        return _commit_compound_variant(best_variant)

    if coe_compound_shortlist_enabled and coe_compound_shortlist:
        selected_variant, coe_reason, coe_summary = _stageA_compound_shortlist_committee_rank(
            base_model=model,
            candidates=coe_compound_shortlist,
            lm_hp=lm_hp,
            y_op=y_op,
            y_op_inv=y_op_inv,
            dtype=dtype,
            device=device,
            data_hp=None,
        )
        print("\n" + _format_stageA_compound_shortlist_committee_report(coe_summary))
        if selected_variant is None:
            print(f"{YELLOW}[CoE StageA compound] {coe_reason}; keeping current AST.{RESET}")
            return False, None, None, None, False, False
        selected_variant = dict(selected_variant)
        selected_variant["coe_stageA_compound_shortlist"] = coe_summary
        print(f"[CoE StageA compound] {coe_reason}")
        return _commit_compound_variant(selected_variant)

    return False, None, None, None, False, False

__search_definitions__ = (
    "_stageA_set_r1_certificate_poly",
    "_try_stageA_r1_operator_certificate_candidates",
    "_stageA_model_reuse_by_tag",
    "_stageA_terminal_closure_rejection_reason",
    "_try_stageA_terminal_closure_probe",
    "_try_stageA_compound_during_sep_for_atom",
    "_try_compound_candidates_for_atom",
)

__search_constants__ = (

)

__search_late_bindings__ = (
    "_accept_threshold_with_structural_target",
    "_format_stageA_compound_shortlist_committee_report",
    "_nn_split_signature",
    "_stageA_compound_shortlist_committee_rank",
    "_stageA_loss_budget_multiplier",
    "_stageA_terminal_closure_committee_gate",
    "_stageA_under_protest_threshold_cap",
)
