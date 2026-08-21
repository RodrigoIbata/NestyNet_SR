# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""
Thin wrappers around the nestynet LM optimiser for the two training regimes:
(1) initial fit of a model,
(2) candidate model fit for a proposed separation.
"""

# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2025 Rodrigo Ibata

import hashlib
import math
from typing import Any, Optional

import nestynet
import torch

from nestynet_sr.sr_core.bridges import AtomNode, eval_input_expr, eval_inputs

from .callbacks import SRCallback, SRState


def _sr_model_state_fingerprint(model) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()

# Project-wide LMConfig overrides (applied to every LMConfig created by SR).
# Centralised here so they can be tuned without touching NestyNet defaults.
SR_LM_OVERRIDES: dict = dict(
    lbfgs_escape_enable=True,
    lbfgs_warm_enable=True,
    direct_predictor_enable=True,
    local_manifold_enable=True,
)


def build_sr_evidence_config(lm_hp, *, epochs: int | None = None) -> Any | None:
    """
    Build an EvidenceConfig from SR hyper-parameters.

    SR keeps model selection plain-chi-square driven. Evidence can guide the
    optimizer underneath, but residual-whitening terms are deliberately rejected
    here because they change the metric in a way that only helps niche low-N
    regimes. Segment priors are allowed, with an optional decay schedule.
    """
    if lm_hp is None or not bool(getattr(lm_hp, "evidence_enable", False)):
        return None

    lambda_patch = float(getattr(lm_hp, "evidence_lambda_patch", 0.0))
    lambda_mean = getattr(lm_hp, "evidence_lambda_mean", None)
    lambda_slope = getattr(lm_hp, "evidence_lambda_slope", None)
    lambda_quad = getattr(lm_hp, "evidence_lambda_quad", None)
    segment_alpha_init = float(getattr(lm_hp, "evidence_segment_alpha_init", 1.0))
    prior_rel_scale = float(getattr(lm_hp, "evidence_prior_rel_scale", 0.25))
    prior_abs_scale = float(getattr(lm_hp, "evidence_prior_abs_scale", 1.0e-3))

    if bool(getattr(lm_hp, "evidence_disable_residual_whitening", False)):
        lambda_patch = 0.0
        lambda_mean = 0.0
        lambda_slope = 0.0
        lambda_quad = 0.0

    if bool(getattr(lm_hp, "evidence_disable_segment_priors", False)):
        segment_alpha_init = 0.0
        prior_rel_scale = 0.0
        prior_abs_scale = 0.0

    patch_terms = [lambda_patch]
    for value in (lambda_mean, lambda_slope, lambda_quad):
        if value is not None:
            patch_terms.append(float(value))

    patch_active = any(abs(float(v)) > 0.0 for v in patch_terms)
    prior_active = abs(float(segment_alpha_init)) > 0.0

    if patch_active:
        raise ValueError(
            "SR evidence mode no longer admits residual-whitening / patch terms. "
            "Use segment priors with decay, or set evidence_disable_residual_whitening=True "
            "and keep evidence_lambda_patch / evidence_lambda_mean / evidence_lambda_slope / "
            "evidence_lambda_quad at zero."
        )

    if not prior_active:
        return None

    decay_start_raw = getattr(lm_hp, "evidence_prior_decay_start", None)
    decay_interval_raw = getattr(lm_hp, "evidence_prior_decay_interval", None)
    decay_start = None if decay_start_raw is None else max(0, int(decay_start_raw))
    decay_interval = 200 if decay_interval_raw is None else max(0, int(decay_interval_raw))

    if decay_start is None and bool(getattr(lm_hp, "evidence_prior_decay_auto", True)):
        default_start = 800
        budget = epochs if epochs is not None else getattr(lm_hp, "epochs", None)
        if budget is not None:
            budget_i = max(1, int(budget))
            max_start_for_full_interval = max(0, int(budget_i - min(decay_interval, budget_i)))
            decay_start = min(default_start, max_start_for_full_interval)
        else:
            decay_start = default_start

    decay_end = None
    if decay_start is not None:
        decay_end = int(decay_start + decay_interval)
        if epochs is not None:
            budget_i = max(1, int(epochs))
            if decay_start > budget_i:
                decay_start = budget_i
            decay_end = min(int(decay_end), budget_i)
            decay_end = max(int(decay_start), int(decay_end))

    return nestynet.optimizer.EvidenceConfig(
        enabled=True,
        objective_normalization="data_mean",
        patch_include_mean=False,
        patch_include_slope=False,
        patch_include_quad=False,
        lambda_patch=0.0,
        lambda_mean=0.0,
        lambda_slope=0.0,
        lambda_quad=0.0,
        segment_alpha_init=segment_alpha_init,
        prior_rel_scale=prior_rel_scale,
        prior_abs_scale=prior_abs_scale,
        prior_decay_start_iter=decay_start,
        prior_decay_end_iter=decay_end,
        prior_decay_shape=str(getattr(lm_hp, "evidence_prior_decay_shape", "cosine") or "cosine"),
        prior_decay_final_scale=float(getattr(lm_hp, "evidence_prior_decay_final_scale", 0.0)),
        log_every_accepted=0,
        update_alpha_every_accepted=0,
        # SR evidence is prior-only here; the optimizer evaluates any
        # refinement candidate on the full augmented objective before accepting.
        allow_linear_refinement=True,
    )


_SR_EVIDENCE_FALLBACK_WARNED: set[tuple[str, str]] = set()


def _sr_evidence_fallback_reason(model) -> tuple[str, str]:
    """Return (category, detail) for an evidence-incompatible SR fit provider."""
    base_model = getattr(model, "base_model", None)
    if base_model is not None and not hasattr(base_model, "num_segments"):
        return "base_without_segments", "top-level base_model has no num_segments"

    leaves = getattr(model, "leaf", None)
    if leaves is not None:
        parts = []
        n_segmented = 0
        n_trainable_unsegmented = 0
        for leaf in list(leaves):
            try:
                npar = int(leaf.num_parameters())
            except Exception:
                try:
                    npar = int(sum(p.numel() for p in leaf.parameters() if p.requires_grad))
                except Exception:
                    npar = -1
            leaf_base = getattr(leaf, "base_model", None)
            is_segmented = leaf_base is not None and hasattr(leaf_base, "num_segments")
            if is_segmented:
                n_segmented += 1
            elif npar > 0:
                n_trainable_unsegmented += 1
            suffix = ":seg" if is_segmented else ""
            parts.append(f"{type(leaf).__name__}:{npar}p{suffix}")
        detail = (
            f"no top-level segmented base_model; leaves={parts}, "
            f"segmented_leaves={n_segmented}, "
            f"trainable_unsegmented_leaves={n_trainable_unsegmented}"
        )
        return "ast_no_top_level_segmented_base", detail

    return "no_base_model", "provider exposes no base_model"


def _validate_sr_evidence_provider(model, evidence_cfg):
    """Return a usable SR evidence config, or collapse to plain LM for unsupported fits."""
    if evidence_cfg is None:
        return None
    base_model = getattr(model, "base_model", None)
    if base_model is None or not hasattr(base_model, "num_segments"):
        category, detail = _sr_evidence_fallback_reason(model)
        warn_key = (type(model).__name__, category)
        if warn_key not in _SR_EVIDENCE_FALLBACK_WARNED:
            _SR_EVIDENCE_FALLBACK_WARNED.add(warn_key)
            print(
                "[SR evidence] "
                f"Disabling evidence for unsupported fit provider {type(model).__name__} "
                f"({detail}); falling back to plain LM for matching fits. "
                "Further identical fallback notices are suppressed."
            )
        return None
    return evidence_cfg

def freeze_non_nn_leaves(model, ast_root):
    """Set requires_grad_(False) on all non-NN leaf parameters.

    This allows Stage A training to update only NN leaves while preserving
    analytical leaves discovered by Stage B.

    Returns the list of frozen parameters so they can be unfrozen later.
    """
    from nestynet_sr.sr_search.stageB.atom_mapping import _collect_all_atoms

    atoms = _collect_all_atoms(ast_root)
    frozen = []
    for atom, leaf in zip(atoms, list(model.leaf)):
        if isinstance(atom, AtomNode) and atom.kind.lower() != "nn":
            for p in leaf.parameters():
                if p.requires_grad:
                    p.requires_grad_(False)
                    frozen.append(p)
    return frozen


def unfreeze_params(params):
    """Restore requires_grad on previously frozen parameters."""
    for p in params:
        p.requires_grad_(True)


def _sr_lm_iter_check(lm_opt) -> int:
    """Return the LM validation/report cadence used by SR."""
    try:
        iter_check = int(getattr(lm_opt, "iter_check", 0) or 0)
    except Exception:
        iter_check = 0
    return max(iter_check, 1)


def _sr_evidence_metrics_ready(lm_opt) -> bool:
    """Gate SR decisions until evidence segment-prior decay has finished."""
    evidence_cfg = getattr(lm_opt, "evidence_cfg", None)
    if evidence_cfg is None:
        return True
    if not bool(getattr(evidence_cfg, "enabled", False)):
        return True
    lm_hp_gate = bool(getattr(evidence_cfg, "sr_gate_metrics_until_prior_decay", True))
    if not lm_hp_gate:
        return True
    ctrl = getattr(lm_opt, "evidence_controller", None)
    if ctrl is None or not hasattr(ctrl, "prior_decay_enabled"):
        return True
    try:
        if not bool(ctrl.prior_decay_enabled()):
            return True
        return bool(ctrl.prior_decay_complete())
    except Exception:
        return True


def _sr_validation_fresh_after_step(
    lm_opt,
    *,
    require_metrics_ready: bool = True,
) -> bool:
    """Whether ``loss_val`` returned by the latest ``step()`` is freshly evaluated."""
    if getattr(lm_opt, "datagen_val", None) is None:
        return False
    if require_metrics_ready and not _sr_evidence_metrics_ready(lm_opt):
        return False
    state = getattr(lm_opt, "state", {}) or {}
    if bool(state.get("halt", False)):
        return True
    iter_check = _sr_lm_iter_check(lm_opt)
    try:
        it_now = int(state.get("iter", 0))
    except Exception:
        it_now = 0
    it_eval = it_now - 1
    return bool(it_eval == 0 or (it_eval > 0 and it_eval % iter_check == 0))


def _sr_report_iteration_after_step(lm_opt) -> int | None:
    """Return the LM report iteration that just completed, or ``None``."""
    state = getattr(lm_opt, "state", {}) or {}
    iter_check = _sr_lm_iter_check(lm_opt)
    try:
        it_now = int(state.get("iter", 0))
    except Exception:
        it_now = 0
    it_eval = it_now - 1
    if it_eval == 0 or (it_eval > 0 and it_eval % iter_check == 0):
        return int(it_eval)
    return None


def _sr_metric_float(loss_metrics: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        raw = loss_metrics.get(key, None)
        if raw is None:
            continue
        try:
            value = float(raw)
        except Exception:
            continue
        if math.isfinite(value):
            return value
    return None


def _sr_plain_train_selection_loss(loss_metrics: dict[str, Any]) -> float | None:
    """Return the canonical SR early-cutoff metric on the SR selection scale."""
    return _sr_metric_float(
        loss_metrics,
        "train_selection_loss",
        "train_data_target_mean_sum_loss",
        "train_data_mean_loss",
        "train_data_sum_loss",
    )


def _sr_maybe_trigger_prior_decay_from_stall(
    lm_opt,
    *,
    loss_metrics: dict[str, Any],
    prev_report_train_selection_loss: float | None,
    epochs: int | None = None,
    label: str = "",
) -> tuple[float | None, bool]:
    """
    Start prior decay early when plain train selection-loss improvement stalls.

    Under SR evidence the augmented optimizer objective is mean-normalized, so
    this trigger lives on the same scale as the usual SR acceptance and patience
    thresholds.
    """
    report_iter = _sr_report_iteration_after_step(lm_opt)
    current_selection = _sr_plain_train_selection_loss(loss_metrics)
    if report_iter is None or current_selection is None:
        return prev_report_train_selection_loss, False
    if _sr_lm_halt_is_lam_max_rejects(lm_opt):
        return current_selection, False

    evidence_cfg = getattr(lm_opt, "evidence_cfg", None)
    if evidence_cfg is None or not bool(getattr(evidence_cfg, "enabled", False)):
        return current_selection, False

    cutoff_tol = getattr(evidence_cfg, "sr_prior_cutoff_tol", None)
    if cutoff_tol is None:
        return current_selection, False

    ctrl = getattr(lm_opt, "evidence_controller", None)
    if ctrl is None or not hasattr(ctrl, "prior_decay_enabled"):
        return current_selection, False
    try:
        if not bool(ctrl.prior_decay_enabled()) or bool(ctrl.prior_decay_complete()):
            return current_selection, False
    except Exception:
        return current_selection, False

    ctrl_cfg = getattr(ctrl, "cfg", None)
    if ctrl_cfg is None:
        return current_selection, False

    start_raw = getattr(ctrl_cfg, "prior_decay_start_iter", None)
    end_raw = getattr(ctrl_cfg, "prior_decay_end_iter", None)
    if start_raw is None and end_raw is None:
        return current_selection, False
    start_i = 0 if start_raw is None else max(0, int(start_raw))
    end_i = start_i if end_raw is None else max(0, int(end_raw))
    next_iter = int(report_iter) + 1
    if next_iter >= start_i:
        return current_selection, False

    if prev_report_train_selection_loss is None:
        return current_selection, False

    improvement = float(prev_report_train_selection_loss) - float(current_selection)
    lm_opt.state["sr_prior_cutoff_metric"] = "train_selection_loss"
    lm_opt.state["sr_prior_cutoff_last_improvement"] = float(improvement)
    lm_opt.state["sr_prior_cutoff_tol"] = float(cutoff_tol)

    if not math.isfinite(improvement):
        return current_selection, False
    if improvement < 0.0:
        return current_selection, False
    if improvement >= float(cutoff_tol):
        return current_selection, False

    duration = getattr(evidence_cfg, "sr_prior_decay_duration", None)
    if duration is None:
        duration = max(0, int(end_i - start_i))
    duration_i = max(0, int(duration))
    new_start = int(next_iter)
    new_end = int(new_start + duration_i)
    if epochs is not None:
        budget_i = max(1, int(epochs))
        new_start = min(new_start, budget_i)
        new_end = min(new_end, budget_i)
    new_end = max(new_start, new_end)

    ctrl_cfg.prior_decay_start_iter = int(new_start)
    ctrl_cfg.prior_decay_end_iter = int(new_end)
    lm_opt.state["sr_prior_decay_trigger"] = {
        "kind": "plain_selection_stall",
        "metric": "train_selection_loss",
        "report_iter": int(report_iter),
        "next_iter": int(next_iter),
        "improvement": float(improvement),
        "tol": float(cutoff_tol),
        "new_start_iter": int(new_start),
        "new_end_iter": int(new_end),
    }
    print(
        f"{label}[SR evidence] Early prior decay trigger at report iter {int(report_iter)}: "
        f"plain-data selection-loss improvement {float(improvement):.6e} < cutoff {float(cutoff_tol):.6e}; "
        f"rescheduling decay {int(new_start)}->{int(new_end)}."
    )
    return current_selection, True


def _sr_maybe_pull_forward_prior_decay_on_halt(
    lm_opt,
    *,
    epochs: int | None = None,
    label: str = "",
) -> bool:
    """
    If LM wants to halt before prior decay has started, start decay next iter.

    This avoids the silly SR evidence sequence where LM has already reached a
    halt condition on the current objective, but SR then keeps deferring the
    halt while waiting for a much later fixed decay start.
    """
    if _sr_lm_halt_is_lam_max_rejects(lm_opt):
        return False

    evidence_cfg = getattr(lm_opt, "evidence_cfg", None)
    if evidence_cfg is None or not bool(getattr(evidence_cfg, "enabled", False)):
        return False

    ctrl = getattr(lm_opt, "evidence_controller", None)
    if ctrl is None or not hasattr(ctrl, "prior_decay_enabled"):
        return False
    try:
        if not bool(ctrl.prior_decay_enabled()) or bool(ctrl.prior_decay_complete()):
            return False
    except Exception:
        return False

    ctrl_cfg = getattr(ctrl, "cfg", None)
    if ctrl_cfg is None:
        return False

    start_raw = getattr(ctrl_cfg, "prior_decay_start_iter", None)
    end_raw = getattr(ctrl_cfg, "prior_decay_end_iter", None)
    if start_raw is None and end_raw is None:
        return False
    start_i = 0 if start_raw is None else max(0, int(start_raw))
    end_i = start_i if end_raw is None else max(0, int(end_raw))

    state = getattr(lm_opt, "state", {}) or {}
    try:
        next_iter = max(0, int(state.get("iter", 0)))
    except Exception:
        next_iter = 0
    if next_iter >= start_i:
        return False

    duration = getattr(evidence_cfg, "sr_prior_decay_duration", None)
    if duration is None:
        duration = max(0, int(end_i - start_i))
    duration_i = max(0, int(duration))
    new_start = int(next_iter)
    new_end = int(new_start + duration_i)
    if epochs is not None:
        budget_i = max(1, int(epochs))
        new_start = min(new_start, budget_i)
        new_end = min(new_end, budget_i)
    new_end = max(new_start, new_end)

    ctrl_cfg.prior_decay_start_iter = int(new_start)
    ctrl_cfg.prior_decay_end_iter = int(new_end)
    lm_opt.state["sr_prior_decay_trigger"] = {
        "kind": "lm_pre_decay_halt",
        "metric": "lm_halt",
        "next_iter": int(next_iter),
        "new_start_iter": int(new_start),
        "new_end_iter": int(new_end),
    }
    print(
        f"{label}[SR evidence] LM requested halt before prior decay started; "
        f"rescheduling decay {int(new_start)}->{int(new_end)}."
    )
    return True


def _sr_lm_halt_is_lam_max_rejects(lm_opt) -> bool:
    """Return True for the LM terminal halt caused by repeated rejects at lam_max."""
    state = getattr(lm_opt, "state", {}) or {}
    try:
        cap_rejects = int(state.get("lam_at_cap_rejects", 0) or 0)
    except Exception:
        return False
    try:
        max_cap_rejects = int(getattr(lm_opt, "max_lam_at_cap_rejects", 0) or 0)
    except Exception:
        max_cap_rejects = 0
    if max_cap_rejects <= 0 or cap_rejects < max_cap_rejects:
        return False

    lam_max = getattr(lm_opt, "lam_max", None)
    lam = state.get("lam", None)
    if lam_max is None or lam is None:
        return True
    try:
        lam_max_f = float(lam_max)
        lam_f = float(lam)
    except Exception:
        return True
    if not math.isfinite(lam_max_f) or lam_max_f <= 0.0:
        return True
    if not math.isfinite(lam_f):
        return False
    return bool(lam_f >= (1.0 - 1.0e-12) * lam_max_f)


def _sr_latest_single_target_loss_metrics(lm_opt, *, label: str = "") -> dict[str, Any]:
    """
    Return plain train/val data metrics for the common single-target SR LM paths.

    ``train_selection_loss`` and ``val_selection_loss`` are the canonical SR
    comparison scale. For a single target that is just the ordinary data mean,
    irrespective of whether evidence has augmented the optimizer objective.
    """
    metrics = dict(lm_opt.latest_loss_metrics(target_module_index=0))
    if bool(metrics.get("evidence_active", False)):
        val_modules = list(getattr(lm_opt, "residual_modules_val", None) or [])
        if len(val_modules) > 1:
            prefix = label if label else ""
            raise NotImplementedError(
                f"{prefix}Evidence mode is not yet supported for joint LM fits "
                f"with {len(val_modules)} validation target residual modules. "
                "SR needs an explicit plain-data aggregation policy there."
            )

    train_mean = metrics.get("train_data_mean_loss", None)
    val_mean = metrics.get("val_data_mean_loss", None)
    if train_mean is not None:
        metrics.setdefault("train_selection_loss", float(train_mean))
        metrics.setdefault("train_data_target_mean_sum_loss", float(train_mean))
        metrics.setdefault("train_data_global_mean_loss", float(train_mean))
    if val_mean is not None:
        metrics.setdefault("val_selection_loss", float(val_mean))
        metrics.setdefault("val_data_target_mean_sum_loss", float(val_mean))
        metrics.setdefault("val_data_global_mean_loss", float(val_mean))
    return metrics


def _sr_latest_joint_loss_metrics(
    lm_opt,
    *,
    target_count: int | None = None,
    label: str = "",
) -> dict[str, Any]:
    """
    Return plain-data train/val metrics for joint multi-target SR LM fits.

    The canonical SR comparison scale is the historical one: sum of per-target
    mean losses. True residual sums and global residual means are reported as
    diagnostics, but they are not silently substituted for model selection.
    """
    base_modules = list(
        getattr(lm_opt, "base_residual_modules", None)
        or getattr(lm_opt, "residual_modules", None)
        or []
    )
    val_modules = list(getattr(lm_opt, "residual_modules_val", None) or [])

    if target_count is None:
        target_count = len(val_modules) if val_modules else len(base_modules)
    target_count = int(target_count)
    if target_count <= 0:
        raise ValueError(f"{label}Joint loss metrics require at least one target module.")
    if base_modules and target_count > len(base_modules):
        raise ValueError(
            f"{label}Requested {target_count} joint targets, but optimizer only has "
            f"{len(base_modules)} base residual modules."
        )
    if val_modules and target_count > len(val_modules):
        raise ValueError(
            f"{label}Requested {target_count} joint targets, but optimizer only has "
            f"{len(val_modules)} validation residual modules."
        )

    train_selection = 0.0
    val_selection = 0.0 if val_modules else None
    train_sum_total = 0.0
    val_sum_total = 0.0 if val_modules else None
    train_n_total = 0
    val_n_total = 0
    train_sum_available = True
    val_sum_available = bool(val_modules)
    evidence_active = False
    objective_is_augmented = False
    per_target = []

    for idx in range(target_count):
        metrics = dict(lm_opt.latest_loss_metrics(target_module_index=idx))
        evidence_active = evidence_active or bool(metrics.get("evidence_active", False))
        objective_is_augmented = objective_is_augmented or bool(
            metrics.get("objective_is_augmented", False)
        )

        train_loss = metrics.get("train_data_mean_loss", None)
        if train_loss is None:
            raise RuntimeError(
                f"{label}Missing plain training loss for joint target module {idx}."
            )
        train_selection += float(train_loss)

        train_sum = metrics.get("train_data_sum_loss", None)
        train_n = metrics.get("train_n_residuals", None)
        if train_sum is None or train_n is None:
            train_sum_available = False
        else:
            train_sum_total += float(train_sum)
            train_n_total += int(train_n)

        val_loss = metrics.get("val_data_mean_loss", None)
        if val_selection is not None:
            if val_loss is None:
                raise RuntimeError(
                    f"{label}Missing plain validation loss for joint target module {idx}."
                )
            val_selection += float(val_loss)

            val_sum = metrics.get("val_data_sum_loss", None)
            val_n = metrics.get("val_n_residuals", None)
            if val_sum is None or val_n is None:
                val_sum_available = False
            else:
                val_sum_total += float(val_sum)
                val_n_total += int(val_n)

        metrics.setdefault("train_selection_loss", float(train_loss))
        metrics.setdefault("train_data_target_mean_sum_loss", float(train_loss))
        if val_loss is not None:
            metrics.setdefault("val_selection_loss", float(val_loss))
            metrics.setdefault("val_data_target_mean_sum_loss", float(val_loss))
        per_target.append(metrics)

    train_true_sum = float(train_sum_total) if train_sum_available else None
    val_true_sum = float(val_sum_total) if val_sum_available else None
    train_global_mean = (
        float(train_sum_total) / float(train_n_total)
        if train_sum_available and train_n_total > 0 else None
    )
    val_global_mean = (
        float(val_sum_total) / float(val_n_total)
        if val_sum_available and val_n_total > 0 else None
    )

    return {
        "target_module_count": target_count,
        "evidence_active": evidence_active,
        "objective_is_augmented": objective_is_augmented,
        "train_selection_loss": float(train_selection),
        "val_selection_loss": None if val_selection is None else float(val_selection),
        "train_data_target_mean_sum_loss": float(train_selection),
        "val_data_target_mean_sum_loss": None if val_selection is None else float(val_selection),
        "train_data_mean_loss": float(train_selection),
        "val_data_mean_loss": None if val_selection is None else float(val_selection),
        "train_data_sum_loss": train_true_sum,
        "val_data_sum_loss": val_true_sum,
        "train_n_residuals": int(train_n_total) if train_sum_available else None,
        "val_n_residuals": int(val_n_total) if val_sum_available else None,
        "train_data_global_mean_loss": train_global_mean,
        "val_data_global_mean_loss": val_global_mean,
        "per_target": per_target,
    }


def _sr_align_validation_patience(nval_patience: int, iter_check: int, label: str = "") -> int:
    """Align validation patience to the sparse validation cadence."""
    patience = int(nval_patience)
    if patience <= 0:
        raise ValueError(f"{label}nval_patience must be positive, got {patience}.")
    if iter_check <= 1 or patience % iter_check == 0:
        return patience
    aligned = int(iter_check * math.ceil(float(patience) / float(iter_check)))
    print(
        f"{label}Adjusting nval_patience from {patience} to {aligned} "
        f"to align with iter_check={iter_check}."
    )
    return aligned


def _sr_warn_validation_epoch_alignment(epoch_target, iter_check: int, name: str, label: str = "") -> None:
    """Warn when a validation-based epoch threshold falls between validation reports."""
    if epoch_target is None or iter_check <= 1:
        return
    try:
        epoch_v = int(epoch_target)
    except Exception:
        return
    if epoch_v < 0 or epoch_v % iter_check == 0:
        return
    aligned = int(iter_check * math.ceil(float(epoch_v) / float(iter_check)))
    print(
        f"{label}{name}={epoch_v} does not align with iter_check={iter_check}; "
        f"the validation-based check will run at the next fresh validation epoch ({aligned})."
    )


def _make_lm_optimizer(
    model,
    train_dl,
    val_dl,
    LM_strategy,
    chisq_tol,
    device,
    verbose=False,
    log_file=None,
    log_to_console=True,
    log_level=None,
    extra_train_factories=None,
    evidence_cfg=None,
):
    params = list(model.parameters())
    evidence_cfg = _validate_sr_evidence_provider(model, evidence_cfg)

    def seg_factory(dataloader):
        def factory(_):
            return nestynet.optimizer.ResidualsModule(
                providers=[model], dataloader=dataloader, device=device
            )

        return factory

    residual_module_factories = [seg_factory(train_dl)]
    if extra_train_factories:
        residual_module_factories.extend(extra_train_factories)
    residual_module_factories_val = [seg_factory(val_dl)]
    cfg = nestynet.optimizer.LMConfig(
        verbose=verbose,
        LM_strategy=LM_strategy,
        chisq_tol=chisq_tol,
        log_file=log_file,
        log_to_console=log_to_console,
        log_level=log_level,
        **SR_LM_OVERRIDES,
    )
    lm_opt = nestynet.optimizer.Predictive_LM_Optimizer(
        params,
        residual_module_factories,
        residual_module_factories_val=residual_module_factories_val,
        cfg=cfg,
        evidence_cfg=evidence_cfg,
    )
    return lm_opt


def _run_lm_loop(
    lm_opt,
    epochs,
    epochs_min,
    nval_patience,
    loss_target=None,
    accept_threshold=None,
    epochs_awful_check=None,
    awful_threshold=None,
    label="",
    track_params=False,
    callback: Optional[SRCallback] = None,
    state: Optional[SRState] = None,
):
    best_val_loss = float("inf")
    best_train_loss = float("inf")  # Training loss at best validation epoch
    best_param_vec = None
    nval_worse = 0
    halt_reason = None
    # Pre-bound so the post-loop tail is safe if the very first lm_opt.step()
    # raises and we break before assigning these inside the loop body.
    loss = float("inf")
    loss_val = None
    iter_check = _sr_lm_iter_check(lm_opt)
    nval_patience = _sr_align_validation_patience(nval_patience, iter_check, label=label)
    _sr_warn_validation_epoch_alignment(epochs_min, iter_check, "epochs_min", label=label)
    _sr_warn_validation_epoch_alignment(
        epochs_awful_check, iter_check, "epochs_awful_check", label=label
    )
    last_val_epoch = None
    awful_checked = False
    abandon_checked = False
    last_report_train_selection_loss = None

    for epoch in range(epochs + 1):
        try:
            loss_obj, loss_val_obj = lm_opt.step()
        except (RuntimeError, FloatingPointError) as exc:
            # A pathological candidate (e.g. exp(x/cos(...)) overflowing to inf
            # on part of the domain) can poison the LM normal equations so that
            # every dense-solve fallback fails and the optimizer raises. Treat
            # this as a degenerate fit: halt gracefully and return the
            # best-so-far, so the candidate is rejected on its loss like any
            # other rather than crashing the whole run.
            halt_reason = f"LM step failed at epoch {epoch}: {type(exc).__name__}: {exc}"
            print(f"{label}[SR training] {halt_reason}; halting fit and returning best-so-far")
            break
        loss_metrics = _sr_latest_single_target_loss_metrics(
            lm_opt, label="[SR training] "
        )
        loss = float(loss_metrics.get("train_selection_loss", loss_metrics.get("train_data_mean_loss", loss_obj)))
        raw_val = loss_metrics.get("val_selection_loss", loss_metrics.get("val_data_mean_loss", loss_val_obj))
        loss_val = None if raw_val is None else float(raw_val)
        last_report_train_selection_loss, _ = _sr_maybe_trigger_prior_decay_from_stall(
            lm_opt,
            loss_metrics=loss_metrics,
            prev_report_train_selection_loss=last_report_train_selection_loss,
            epochs=epochs,
            label=label,
        )
        metrics_ready = _sr_evidence_metrics_ready(lm_opt)
        visible_loss_val = loss_val if metrics_ready else None
        val_fresh_data = _sr_validation_fresh_after_step(
            lm_opt,
            require_metrics_ready=False,
        )
        val_fresh = _sr_validation_fresh_after_step(lm_opt)

        # Evidence segment priors are optimizer guidance. Model selection must
        # still checkpoint the all-time best plain data-validation loss, even
        # before prior decay has completed.
        if val_fresh_data and loss_val is not None and loss_val < best_val_loss:
            best_val_loss = float(loss_val)
            best_train_loss = float(loss)  # Track training loss at best validation
            if track_params:
                best_param_vec = torch.cat([p.view(-1) for p in lm_opt.params]).detach()

        if val_fresh and visible_loss_val is not None and visible_loss_val <= best_val_loss:
            nval_worse = 0
            last_val_epoch = epoch
        elif val_fresh and visible_loss_val is not None and visible_loss_val > best_val_loss:
            if last_val_epoch is None:
                nval_worse += iter_check
            else:
                nval_worse += max(1, epoch - last_val_epoch)
            last_val_epoch = epoch
        elif val_fresh:
            last_val_epoch = epoch

        # Update state and trigger callback
        if state is not None:
            state.lm_epoch = epoch
            state.lm_max_epochs = epochs
            state.current_loss = float(loss)
            state.current_val_loss = float(visible_loss_val) if visible_loss_val is not None else float("inf")
            state.best_val_loss = min(state.best_val_loss, best_val_loss)

        if callback is not None and state is not None:
            callback.on_lm_epoch(
                state=state,
                epoch=epoch,
                loss=float(loss),
                val_loss=float(visible_loss_val) if visible_loss_val is not None else float("inf"),
            )

        halt_now = bool(lm_opt.state.get("halt", False))
        if halt_now and not metrics_ready:
            if _sr_lm_halt_is_lam_max_rejects(lm_opt):
                lm_opt.state["sr_pre_decay_terminal_halt"] = {
                    "kind": "lam_max_rejects",
                    "iter": int(lm_opt.state.get("iter", -1)),
                    "lam_at_cap_rejects": int(
                        lm_opt.state.get("lam_at_cap_rejects", 0) or 0
                    ),
                }
                if loss_val is not None and loss_val < best_val_loss:
                    best_val_loss = float(loss_val)
                    best_train_loss = float(loss)
                    if track_params:
                        best_param_vec = torch.cat(
                            [p.view(-1) for p in lm_opt.params]
                        ).detach()
                    visible_loss_val = float(loss_val)
                    if state is not None:
                        state.current_val_loss = float(loss_val)
                        state.best_val_loss = min(state.best_val_loss, best_val_loss)
            else:
                pulled_forward = _sr_maybe_pull_forward_prior_decay_on_halt(
                    lm_opt,
                    epochs=epochs,
                    label=label,
                )
                if not pulled_forward and not bool(
                    lm_opt.state.get("_sr_deferred_halt_until_metrics_ready", False)
                ):
                    print(
                        f"{label}Deferring LM halt until evidence prior decay completes;"
                        f" iter={int(lm_opt.state.get('iter', -1))}"
                    )
                lm_opt.state["_sr_deferred_halt_until_metrics_ready"] = True
                lm_opt.state["halt"] = False
                halt_now = False

        if halt_now:
            halt_reason = f"LM halt: {lm_opt.state['halt']}"
            print(
                f"{label}Stopping at epoch {epoch + 1}/{epochs},"
                f" lm_opt.state['halt']={lm_opt.state['halt']},"
                f" loss={float(loss):.4e},"
                f" loss_val={float(visible_loss_val) if visible_loss_val is not None else float('nan'):.4e}"
            )
            break

        if metrics_ready and epoch > epochs_min and loss_target is not None and loss < loss_target:
            halt_reason = f"Target loss reached: {float(loss):.4e} < {float(loss_target):.4e}"
            print(
                f"{label}Target loss reached at epoch {epoch + 1}/{epochs},"
                f" loss={float(loss):.4e} < loss_target={float(loss_target):.4e}"
            )
            lm_opt.state["halt"] = True

        if val_fresh and nval_worse >= nval_patience:
            halt_reason = f"Validation patience exceeded: {nval_worse}"
            print(
                f"{label}Validation patience exceeded at epoch {epoch + 1}/{epochs},"
                f" nval_worse={nval_worse}"
            )
            break

        # Early abandonment for awful fits: at epochs_awful_check, abandon if val_loss is awful
        if (
            val_fresh
            and
            epochs_awful_check is not None
            and awful_threshold is not None
            and (not awful_checked)
            and epoch >= epochs_awful_check
        ):
            awful_checked = True
            if best_val_loss > awful_threshold:
                halt_reason = f"Awful fit at epoch {epoch + 1}: val_loss={best_val_loss:.4e} > {awful_threshold:.4e}"
                print(
                    f"{label}Awful fit detected at epoch {epoch + 1}/{epochs}:"
                    f" best_val_loss={best_val_loss:.4e} > awful_threshold={awful_threshold:.4e}"
                )
                break

        # Early abandonment: at epochs_min, check if loss is too high to be worth continuing
        if val_fresh and epoch >= epochs_min and accept_threshold is not None and not abandon_checked:
            abandon_checked = True
            if best_val_loss != float("inf"):
                # Calculate abandonment threshold: if we'd need to run (epochs/epochs_min)x longer
                # and loss is still (epochs/epochs_min)x worse than acceptable, give up
                abandon_threshold = (epochs / epochs_min) * accept_threshold
                if best_val_loss > abandon_threshold:
                    halt_reason = f"Early abandonment at epochs_min: val_loss={best_val_loss:.4e} > {abandon_threshold:.4e}"
                    print(
                        f"{label}Early abandonment at epoch {epoch + 1}/{epochs}:"
                        f" best_val_loss={best_val_loss:.4e} > {abandon_threshold:.4e}"
                        f" ({epochs / epochs_min:.1f}x accept_threshold {accept_threshold:.4e})"
                    )
                    break

    if best_val_loss == float("inf") and loss_val is not None:
        best_val_loss = float(loss_val)
        best_train_loss = float(loss)

    # Decide acceptance based on the *best* validation loss seen.
    accepted = (
        accept_threshold is not None
        and best_val_loss != float("inf")
        and best_val_loss < accept_threshold
    )

    # Trigger completion callback
    if callback is not None and state is not None:
        if halt_reason is None:
            halt_reason = "Max epochs reached"
        state.lm_halt_reason = halt_reason
        callback.on_lm_complete(
            state=state, reason=halt_reason, final_loss=float(loss), final_val_loss=best_val_loss
        )

    return accepted, best_val_loss, best_train_loss, best_param_vec


def _attach_sr_evidence_runtime_flags(evidence_cfg, lm_hp) -> Any | None:
    if evidence_cfg is None:
        return None
    try:
        evidence_cfg.sr_gate_metrics_until_prior_decay = bool(
            getattr(lm_hp, "evidence_gate_metrics_until_prior_decay", True)
        )
    except Exception:
        pass
    try:
        chisq_tol = getattr(lm_hp, "chisq_tol", None)
        cutoff_tol = getattr(lm_hp, "evidence_prior_cutoff_tol", None)
        if cutoff_tol is not None and chisq_tol is not None:
            cutoff_tol = max(float(cutoff_tol), float(chisq_tol))
        evidence_cfg.sr_prior_cutoff_tol = None if cutoff_tol is None else float(cutoff_tol)
        evidence_cfg.sr_suppress_pre_decay_convergence_halt = bool(
            getattr(evidence_cfg, "sr_gate_metrics_until_prior_decay", True)
        )
        duration = getattr(lm_hp, "evidence_prior_decay_interval", None)
        if duration is None:
            start = getattr(evidence_cfg, "prior_decay_start_iter", None)
            end = getattr(evidence_cfg, "prior_decay_end_iter", None)
            if start is None and end is None:
                evidence_cfg.sr_prior_decay_duration = None
            else:
                start_i = 0 if start is None else max(0, int(start))
                end_i = start_i if end is None else max(0, int(end))
                evidence_cfg.sr_prior_decay_duration = max(0, int(end_i - start_i))
        else:
            evidence_cfg.sr_prior_decay_duration = max(0, int(duration))
    except Exception:
        pass
    return evidence_cfg


_SR_CANONICAL_INIT_UNSUPPORTED_WARNED: set[str] = set()


def _sr_canonical_init_provider_and_atom(model):
    """Return the canonical-init provider and its AST input atom, if any."""
    input_atom = None
    leaves = getattr(model, "leaf", None)
    if leaves is not None:
        try:
            leaves_list = list(leaves)
        except Exception:
            leaves_list = []
        ast_root = getattr(model, "ast_root", None)
        if (
            len(leaves_list) == 1
            and str(getattr(ast_root, "kind", "")).lower() == "nn"
        ):
            candidate = leaves_list[0]
            input_atom = ast_root
        else:
            return None, None
    else:
        candidate = model

    base = getattr(candidate, "base_model", None)
    if hasattr(candidate, "canonical_init_greedy") or (
        base is not None and hasattr(base, "canonical_init_greedy")
    ):
        return candidate, input_atom

    stage0 = getattr(candidate, "stage0", None)
    stage1 = getattr(candidate, "stage1", None)
    base0 = getattr(stage0, "base_model", stage0)
    base1 = getattr(stage1, "base_model", stage1)
    if (
        base0 is not None
        and base1 is not None
        and hasattr(base0, "canonical_init_affine_ls")
        and hasattr(base0, "canonical_init_segment_from_residual")
        and hasattr(base1, "canonical_init_greedy")
    ):
        return candidate, input_atom
    return None, None


def _sr_canonical_init_provider(model):
    """Return the provider that can be safely canonical-initialized for SR."""
    provider, _input_atom = _sr_canonical_init_provider_and_atom(model)
    return provider


def _sr_infer_canonical_dtype(provider, lm_opt) -> torch.dtype:
    for obj in (provider, getattr(provider, "base_model", None), lm_opt):
        if obj is None or not hasattr(obj, "parameters"):
            continue
        try:
            for param in obj.parameters():
                return param.dtype
        except Exception:
            continue
    return torch.get_default_dtype()


def _sr_canonical_provider_input_dim(provider) -> int | None:
    stage0 = getattr(provider, "stage0", None)
    if stage0 is not None:
        base0 = getattr(stage0, "base_model", stage0)
        nx = getattr(base0, "Nx_size", None)
        if nx is not None:
            return int(nx)

    base = getattr(provider, "base_model", provider)
    nx = getattr(base, "Nx_size", None)
    if nx is not None:
        return int(nx)
    return None


def _sr_collect_canonical_loader_data(train_dl, *, device, dtype):
    xs, ys, sigmas = [], [], []
    any_sigma = False
    for batch in train_dl:
        if len(batch) == 2:
            x_b, y_b = batch
            sigma_b = None
        else:
            x_b, y_b, sigma_b = batch
        xs.append(torch.as_tensor(x_b, device=device, dtype=dtype))
        ys.append(torch.as_tensor(y_b, device=device, dtype=dtype))
        if sigma_b is None:
            sigmas.append(None)
        else:
            any_sigma = True
            sigmas.append(torch.as_tensor(sigma_b, device=device, dtype=dtype))

    x_train = torch.cat(xs, dim=0)
    y_train = torch.cat(ys, dim=0)
    y_sigma = None
    if any_sigma and all(s is not None for s in sigmas):
        y_sigma = torch.cat(sigmas, dim=0)
    return x_train, y_train, y_sigma


def _sr_canonical_init_data_for_atom(input_atom, provider, train_dl, *, device, dtype):
    """Collect training data and route x through the AST atom's effective inputs."""
    x_train, y_train, y_sigma = _sr_collect_canonical_loader_data(
        train_dl, device=device, dtype=dtype
    )
    x_eff, _, _ = eval_inputs(input_atom, x_train, need_grad=False, need_hess=False)

    expected_nx = _sr_canonical_provider_input_dim(provider)
    if expected_nx is not None and int(x_eff.shape[1]) != int(expected_nx):
        raise ValueError(
            "SR canonical init input routing produced "
            f"{int(x_eff.shape[1])} columns, but provider expects Nx_size={int(expected_nx)} "
            f"for atom {input_atom!r}."
        )

    if y_sigma is None:
        return x_eff, y_train
    return x_eff, y_train, y_sigma


def _sr_canonical_init_config(lm_hp) -> dict[str, Any]:
    return {
        "canonical_init_affine_first": bool(
            getattr(lm_hp, "canonical_init_affine_first", True)
        ),
        "canonical_init_ridge": float(getattr(lm_hp, "canonical_init_ridge", 1.0e-6)),
        "canonical_init_bias_mode": str(
            getattr(lm_hp, "canonical_init_bias_mode", "quantile")
        ),
        "canonical_init_orthogonalize": bool(
            getattr(lm_hp, "canonical_init_orthogonalize", True)
        ),
    }


def _maybe_run_sr_canonical_initialization(model, lm_opt, train_dl, device, lm_hp) -> bool:
    if lm_hp is None or not bool(getattr(lm_hp, "canonical_init", False)):
        return False

    provider, input_atom = _sr_canonical_init_provider_and_atom(model)
    if provider is None:
        key = type(model).__name__
        if key not in _SR_CANONICAL_INIT_UNSUPPORTED_WARNED:
            _SR_CANONICAL_INIT_UNSUPPORTED_WARNED.add(key)
            print(
                "[canonical_init] requested but skipped for non-pure-NN SR model "
                f"{key}; canonical init is applied to pure Stage-A NN teachers only."
            )
        return False

    from nestynet.training_utils.canonical_init import canonical_initialize_model

    print("[canonical_init] initialising pure NN model before LM.")
    kwargs = {"dataloader": train_dl}
    if input_atom is not None:
        dtype = _sr_infer_canonical_dtype(provider, lm_opt)
        kwargs = {
            "data": _sr_canonical_init_data_for_atom(
                input_atom,
                provider,
                train_dl,
                device=device,
                dtype=dtype,
            )
        }
    canonical_initialize_model(
        provider,
        lm_opt,
        device=device,
        config=_sr_canonical_init_config(lm_hp),
        **kwargs,
    )
    return True


def train_initial_model(
    model,
    train_dl,
    val_dl,
    epochs,
    LM_strategy,
    nval_patience,
    loss_target,
    epochs_min,
    chisq_tol,
    device,
    epochs_awful_check=None,
    awful_threshold=None,
    callback: Optional[SRCallback] = None,
    state: Optional[SRState] = None,
    log_file=None,
    log_to_console=True,
    log_level=None,
    lm_verbose=False,
    lm_hp=None,
    evidence_cfg=None,
):
    """
    Run LM on `model` for the initial (or re-)fit of an expression.

    Returns
    -------
    best_val_loss : float
    best_train_loss : float
        Training loss at the epoch with best validation loss.
    best_param_vec: 1D tensor
        Flattened parameter vector corresponding to the best validation loss.
    lm_opt        : Predictive_LM_Optimizer
        The optimiser instance, so the caller can apply best_param_vec via
        lm_opt._update_param_groups(best_param_vec).
    """
    declare_input_dim = getattr(model, "declare_global_input_dim", None)
    if callable(declare_input_dim):
        dataset = getattr(train_dl, "dataset", None)
        while dataset is not None and not hasattr(dataset, "tensors"):
            parent = getattr(dataset, "dataset", None)
            if parent is None or parent is dataset:
                dataset = None
                break
            dataset = parent
        tensors = getattr(dataset, "tensors", ()) if dataset is not None else ()
        x_tensor = tensors[0] if tensors else None
        if torch.is_tensor(x_tensor) and x_tensor.ndim >= 2:
            declare_input_dim(int(x_tensor.shape[-1]))

    if evidence_cfg is None:
        evidence_cfg = build_sr_evidence_config(lm_hp, epochs=epochs)
    evidence_cfg = _attach_sr_evidence_runtime_flags(evidence_cfg, lm_hp)
    canonical_provider = (
        _sr_canonical_init_provider(model)
        if lm_hp is not None and bool(getattr(lm_hp, "canonical_init", False))
        else None
    )
    if evidence_cfg is not None and canonical_provider is not None:
        try:
            evidence_cfg.prior_anchor_mode = "canonical"
        except Exception:
            pass
    lm_opt = _make_lm_optimizer(
        model,
        train_dl,
        val_dl,
        LM_strategy,
        chisq_tol,
        device,
        verbose=lm_verbose,
        log_file=log_file,
        log_to_console=log_to_console,
        log_level=log_level,
        evidence_cfg=evidence_cfg,
    )
    canonical_init_applied = _maybe_run_sr_canonical_initialization(
        model, lm_opt, train_dl, device, lm_hp
    )
    lm_opt._sr_canonical_init_applied = bool(canonical_init_applied)
    lm_opt._sr_canonical_state_fingerprint = (
        _sr_model_state_fingerprint(model) if canonical_init_applied else None
    )
    accepted, best_val_loss, best_train_loss, best_param_vec = _run_lm_loop(
        lm_opt,
        epochs=epochs,
        epochs_min=epochs_min,
        nval_patience=nval_patience,
        loss_target=loss_target,
        accept_threshold=None,
        epochs_awful_check=epochs_awful_check,
        awful_threshold=awful_threshold,
        label="",
        track_params=True,
        callback=callback,
        state=state,
    )
    return best_val_loss, best_train_loss, best_param_vec, lm_opt


def train_candidate_model(
    model,
    train_dl,
    val_dl,
    epochs,
    LM_strategy,
    nval_patience,
    loss_target,
    accept_threshold,
    epochs_min,
    chisq_tol,
    device,
    epochs_awful_check=None,
    awful_threshold=None,
    callback: Optional[SRCallback] = None,
    state: Optional[SRState] = None,
    log_file=None,
    log_to_console=True,
    log_level=None,
    lm_verbose=False,
    extra_train_factories=None,
    lm_hp=None,
    evidence_cfg=None,
):
    """
    Train a candidate model for a proposed separation.

    The loop is stopped if either LM halts, patience is exceeded, or the
    validation loss goes below `accept_threshold`.

    Returns
    -------
    accepted         : bool
        True if validation loss fell below accept_threshold.
    best_val_loss    : float
        Best validation loss seen during training.
    best_train_loss  : float
        Training loss at the epoch with best validation loss.
    best_param_vec   : 1D tensor
        Flattened parameter vector corresponding to the best validation loss.
    lm_opt           : Predictive_LM_Optimizer
        The optimizer instance, so the caller can apply best_param_vec via
        lm_opt._update_param_groups(best_param_vec).
    """
    if evidence_cfg is None:
        evidence_cfg = build_sr_evidence_config(lm_hp, epochs=epochs)
    evidence_cfg = _attach_sr_evidence_runtime_flags(evidence_cfg, lm_hp)
    canonical_provider = (
        _sr_canonical_init_provider(model)
        if lm_hp is not None and bool(getattr(lm_hp, "canonical_init", False))
        else None
    )
    if evidence_cfg is not None and canonical_provider is not None:
        try:
            evidence_cfg.prior_anchor_mode = "canonical"
        except Exception:
            pass
    temp_opt = _make_lm_optimizer(
        model,
        train_dl,
        val_dl,
        LM_strategy,
        chisq_tol,
        device,
        verbose=lm_verbose,
        log_file=log_file,
        log_to_console=log_to_console,
        log_level=log_level,
        extra_train_factories=extra_train_factories,
        evidence_cfg=evidence_cfg,
    )
    canonical_init_applied = _maybe_run_sr_canonical_initialization(
        model, temp_opt, train_dl, device, lm_hp
    )
    temp_opt._sr_canonical_init_applied = bool(canonical_init_applied)
    temp_opt._sr_canonical_state_fingerprint = (
        _sr_model_state_fingerprint(model)
        if canonical_init_applied
        else None
    )
    accepted, best_val_loss, best_train_loss, best_param_vec = _run_lm_loop(
        temp_opt,
        epochs=epochs,
        epochs_min=epochs_min,
        nval_patience=nval_patience,
        loss_target=loss_target,
        accept_threshold=accept_threshold,
        epochs_awful_check=epochs_awful_check,
        awful_threshold=awful_threshold,
        label="[trial] ",
        track_params=True,
        callback=callback,
        state=state,
    )
    return accepted, best_val_loss, best_train_loss, best_param_vec, temp_opt


def pretrain_compound_leaf_from_teacher(
    compound_model,
    original_leaf: torch.nn.Module,
    compound_leaf: torch.nn.Module,
    z_ast,
    x_data: torch.Tensor,
    original_var_idxs,
    device: torch.device,
    dtype: torch.dtype,
    extra_var_idxs=None,
    extra_input_asts=None,
    prefactor_ast=None,
    original_compound_z_ast=None,
    original_compound_extra_idxs=None,
    epochs: int = 2000,
    lr: float = 1e-2,
    target_loss: float = 1e-4,
    verbose: bool = False,
    original_input_asts=None,
):
    """
    Warm-start compound leaf by distilling from original multi-input leaf.

    The original leaf has learned f(x1, x2, ...) and we want the compound
    leaf to learn g(z, x_extra) where z = compound_expr(x_i, x_j, ...) such that
    g(z, x_extra) ≈ f(x). Variables with exponent 0 in the compound are kept as
    extra independent inputs.

    Parameters
    ----------
    compound_model : ASTCompositeAdaptor
        The compound model with the compound leaf.
    original_leaf : torch.nn.Module
        The original multi-input leaf we're replacing.
    compound_leaf : torch.nn.Module
        The compound leaf to initialize.
    z_ast : Node
        AST for the compound variable expression (e.g., x1*x2).
    x_data : torch.Tensor
        Training data [N, Nxvars].
    original_var_idxs : list
        Indices of variables that the original leaf used.
    device : torch.device
    dtype : torch.dtype
    extra_var_idxs : list, optional
        Indices of non-participating variables (exponent=0) to keep as extra inputs.
    extra_input_asts : list, optional
        Additional analytic compound inputs appended after raw extras. This matches
        AtomNode.inputs order for bundled compound moves: [z, raw extras, extra ASTs].
    prefactor_ast : Node, optional
        If provided, distill the compound leaf to the *scaled* target:
            g(z, x_extra) \approx f(x) / m(x)
        where m(x) is the monomial prefactor described by prefactor_ast.
    original_compound_z_ast : Node, optional
        If the original atom was already a compound, this is the z expression
        it uses. The original leaf expects [z_orig, extra1, ...] not raw vars.
    original_compound_extra_idxs : list, optional
        Extra variable indices from the original compound (if any).
    original_input_asts : sequence of Node, optional
        Exact ordered input expressions consumed by ``original_leaf``.  This
        takes precedence over the legacy single-compound/raw-extra arguments
        and preserves multi-compound inputs such as ``[z0, z1]``.
    epochs : int
        Maximum number of distillation epochs.
    lr : float
        Learning rate for Adam.
    target_loss : float
        Early stopping threshold - stop when loss falls below this.
    verbose : bool
        Print progress.

    Returns
    -------
    compound_model : The same model, with compound leaf initialized.
    """
    if extra_var_idxs is None:
        extra_var_idxs = []
    if extra_input_asts is None:
        extra_input_asts = []

    # Evaluate z = compound_expr(x) using bridges.eval_input_expr
    with torch.no_grad():
        z_vals = eval_input_expr(z_ast, x_data)  # [B, 1]

        # Concatenate with extra variables / compound-expression extras if present.
        # Keep the same order as _build_compound_candidate_ast:
        #   [z_ast] + raw extras + extra_input_asts
        student_parts = [z_vals]
        if extra_var_idxs:
            extra_vals = x_data[:, extra_var_idxs]  # [B, num_extra]
            student_parts.append(extra_vals)
        for extra_ast in extra_input_asts:
            extra_ast_vals = eval_input_expr(extra_ast, x_data)
            if extra_ast_vals.dim() == 1:
                extra_ast_vals = extra_ast_vals.view(-1, 1)
            student_parts.append(extra_ast_vals)
        student_input = torch.cat(student_parts, dim=1) if len(student_parts) > 1 else z_vals

        # Get teacher targets: what the original leaf outputs for these inputs.
        # Preserve the exact ordered input expressions when supplied.  Flattening
        # a compound extra such as z1=(x0-x1) into raw [x0, x1] changes both the
        # semantics and arity expected by the existing leaf.
        if original_input_asts is not None:
            teacher_parts = []
            for input_ast in original_input_asts:
                input_values = eval_input_expr(input_ast, x_data)
                if input_values.dim() == 1:
                    input_values = input_values.view(-1, 1)
                if input_values.dim() != 2 or input_values.shape[1] != 1:
                    raise ValueError(
                        "Each original leaf input expression must evaluate to "
                        f"one column; got shape {tuple(input_values.shape)}"
                    )
                teacher_parts.append(input_values)
            if teacher_parts:
                leaf_input = torch.cat(teacher_parts, dim=1)
            else:
                leaf_input = x_data.new_empty((x_data.shape[0], 0))
            y_teacher = original_leaf(leaf_input)
        # Legacy compatibility path: one compound coordinate followed by raw
        # variable extras.
        elif original_compound_z_ast is not None:
            z_orig = eval_input_expr(original_compound_z_ast, x_data)  # [B, 1]
            orig_extra_idxs = original_compound_extra_idxs or []
            if orig_extra_idxs:
                orig_extra_vals = x_data[:, list(orig_extra_idxs)]
                leaf_input = torch.cat([z_orig, orig_extra_vals], dim=1)
            else:
                leaf_input = z_orig
            y_teacher = original_leaf(leaf_input)
        else:
            # Original leaf expects raw variables
            x_subset = x_data[:, original_var_idxs]
            y_teacher = original_leaf(x_subset)  # [B, 1]

        # Optional: if Stage-A peeled a monomial prefactor outside the leaf,
        # then the compound leaf should learn the *residual* g(z, extras) ≈ f(x)/m(x).
        if prefactor_ast is not None:
            try:
                m_vals = eval_input_expr(prefactor_ast, x_data)
                y_teacher = y_teacher / (m_vals + 1e-30)
            except Exception:
                pass

    # Distillation training with early stopping
    optimizer = torch.optim.Adam(compound_leaf.parameters(), lr=lr)

    for epoch in range(epochs):
        y_student = compound_leaf(student_input)
        loss = torch.mean((y_student - y_teacher) ** 2)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if verbose and (epoch == 0 or (epoch + 1) % 1000 == 0):
            print(f"[Distill] epoch {epoch+1}/{epochs}: loss={loss.item():.4e}")

        # Early stopping
        if loss.item() < target_loss:
            if verbose:
                print(f"[Distill] Converged at epoch {epoch+1}: loss={loss.item():.4e}")
            break

    if verbose and loss.item() >= target_loss:
        print(f"[Distill] Final: loss={loss.item():.4e} (did not converge below {target_loss:.4e})")

    return compound_model
