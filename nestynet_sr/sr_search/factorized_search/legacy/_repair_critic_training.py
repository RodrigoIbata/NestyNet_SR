# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

from __future__ import annotations

"""Training entry points for repair-critic and candidate-ranking models."""

import math
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F

from ._repair_critic_features import (
    BUILD_TUPLE_PREVIEW_FEATURE_NAMES,
    REPAIR_CRITIC_BUILD_TUPLE_MODEL_KIND,
    REPAIR_CRITIC_DEFAULT_UTILITY_WEIGHTS,
    REPAIR_CRITIC_FEATURE_NAMES,
    REPAIR_CRITIC_HEAD_NAMES,
    REPAIR_CRITIC_MACRO_ACTION_NAMES,
    REPAIR_CRITIC_MODE_NAMES,
    REPAIR_CRITIC_PATH_FEATURE_NAMES,
    REPAIR_CRITIC_PATH_RELATION_NAMES,
    REPAIR_CRITIC_PREVIEW_FEATURE_NAMES,
    REPAIR_CRITIC_ROUTE_COMPARE_MODEL_KIND,
    REPAIR_CRITIC_ROUTE_NAMES,
    REPAIR_CRITIC_SHARED_CANDIDATE_MODEL_KIND,
    REPAIR_CRITIC_SHARED_MODEL_KIND,
    REPAIR_ROUTE_COMPARE_EXTRA_FEATURE_NAMES,
    UNIFIED_CANDIDATE_PREVIEW_FEATURE_NAMES,
    _build_actor_critic_rows,
    _build_build_tuple_rows,
    _build_oracle_pretrain_rows,
    _build_repair_build_route_rows,
    _build_repair_slate_rows,
    _build_repair_tuple_rows,
    _build_training_rows,
    _build_unified_candidate_rows,
    _normalize_action_name,
    _to_float,
    build_preview_feature_vector,
    repair_path_feature_vector,
    repair_preview_feature_vector,
    unified_candidate_preview_feature_vector,
)

from ._repair_critic_models import (
    _BuildTupleRankerNet,
    _RepairControllerSharedNet,
    _RepairRouteCompareNet,
    _SharedCandidateDualRankerNet,
    _aux_loss_from_logits,
    _aux_predictions_from_logits,
    _binary_classification_metrics,
    _binary_head_pos_weight,
    _compute_feature_stats,
    _flat_classification_metrics,
    _gather_path_action_scores,
    _gather_path_head,
    _hierarchical_macro_metrics,
    _inverse_frequency_class_weights,
    _listwise_slate_loss,
    _macro_metrics_from_logits,
    _masked_pairwise_pairs_from_targets,
    _masked_path_cross_entropy,
    _maybe_init_model_from_bundle,
    _metrics_from_preds,
    _normalize_inputs,
    _pairwise_rank_loss,
    _path_metrics_from_logits,
    _regression_metrics,
    _route_action_mask,
    _route_emergence_metrics,
    _route_masked_action_logits,
    _slate_rank_metrics,
    _split_indices,
)

def train_build_tuple_ranker(
    rows: Sequence[Any],
    *,
    hidden_dim: int = 32,
    epochs: int = 120,
    lr: float = 5.0e-3,
    weight_decay: float = 1.0e-4,
    val_fraction: float = 0.2,
    seed: int = 0,
    listwise_beta: float = 2.0,
    score_gap_floor: float = 1.0e-3,
    pairwise_gap_scale: float = 0.1,
    state_value_weight: float = 0.10,
    init_bundle: dict[str, Any] | None = None,
) -> dict[str, Any]:
    torch.manual_seed(int(seed))
    training_rows = _build_build_tuple_rows(
        rows,
        score_gap_floor=float(score_gap_floor),
    )
    if len(training_rows) < 8:
        raise ValueError("Need at least 8 same-parent build slates to train a build tuple ranker.")

    feature_names = list(REPAIR_CRITIC_FEATURE_NAMES)
    preview_feature_names = list(BUILD_TUPLE_PREVIEW_FEATURE_NAMES)
    n_rows = len(training_rows)
    x = torch.tensor(
        [[float(row["features"].get(name, 0.0)) for name in feature_names] for row in training_rows],
        dtype=torch.float32,
    )
    max_preview = max((len(list(row.get("preview_rows", []) or [])) for row in training_rows), default=0)
    max_preview = max(1, int(max_preview))
    preview_x = torch.zeros((n_rows, max_preview, len(preview_feature_names)), dtype=torch.float32)
    preview_mask = torch.zeros((n_rows, max_preview), dtype=torch.bool)
    y_preview_utility = torch.zeros((n_rows, max_preview), dtype=torch.float32)
    full_slate_mask = torch.zeros((n_rows,), dtype=torch.bool)
    y_state_value = torch.zeros((n_rows,), dtype=torch.float32)
    pairwise_pairs: list[list[tuple[int, int, float]]] = []

    for i, row in enumerate(training_rows):
        preview_rows = list(row.get("preview_rows", []) or [])
        utility_targets = list(row.get("preview_utility_targets", []) or [])
        for j, preview_row in enumerate(preview_rows[:max_preview]):
            preview_mask[i, j] = True
            preview_x[i, j] = torch.tensor(
                build_preview_feature_vector(preview_row, feature_names=preview_feature_names),
                dtype=torch.float32,
            )
            if j < len(utility_targets):
                y_preview_utility[i, j] = float(utility_targets[j])
        full_slate_mask[i] = bool(row.get("full_slate", False))
        y_state_value[i] = float(row.get("state_value_target", 0.0) or 0.0)
        pairwise_pairs.append([
            (int(better_idx), int(worse_idx), float(gap))
            for better_idx, worse_idx, gap in list(row.get("pairwise_pairs", []) or [])
            if int(better_idx) >= 0 and int(worse_idx) >= 0
        ])

    train_idx, val_idx = _split_indices(n_rows, float(val_fraction), int(seed))
    x_train = x[train_idx]
    x_val = x[val_idx] if len(val_idx) > 0 else x_train
    preview_x_train = preview_x[train_idx]
    preview_x_val = preview_x[val_idx] if len(val_idx) > 0 else preview_x_train
    preview_mask_train = preview_mask[train_idx]
    preview_mask_val = preview_mask[val_idx] if len(val_idx) > 0 else preview_mask_train
    y_preview_utility_train = y_preview_utility[train_idx]
    y_preview_utility_val = y_preview_utility[val_idx] if len(val_idx) > 0 else y_preview_utility_train
    full_slate_mask_train = full_slate_mask[train_idx]
    full_slate_mask_val = full_slate_mask[val_idx] if len(val_idx) > 0 else full_slate_mask_train
    y_state_value_train = y_state_value[train_idx]
    y_state_value_val = y_state_value[val_idx] if len(val_idx) > 0 else y_state_value_train
    pairwise_pairs_train = [pairwise_pairs[int(i)] for i in train_idx.tolist()]
    pairwise_pairs_val = [pairwise_pairs[int(i)] for i in val_idx.tolist()] if len(val_idx) > 0 else pairwise_pairs_train

    mean, std = _compute_feature_stats(x_train)
    x_train_n = _normalize_inputs(x_train, mean, std)
    x_val_n = _normalize_inputs(x_val, mean, std)
    if bool(preview_mask_train.any().item()):
        preview_train_flat = preview_x_train[preview_mask_train]
        preview_mean, preview_std = _compute_feature_stats(preview_train_flat)
    elif isinstance(init_bundle, dict):
        preview_mean = torch.as_tensor(
            init_bundle.get("build_preview_feature_mean", torch.zeros(len(preview_feature_names))),
            dtype=torch.float32,
        )
        preview_std = torch.as_tensor(
            init_bundle.get("build_preview_feature_std", torch.ones(len(preview_feature_names))),
            dtype=torch.float32,
        )
    else:
        preview_mean = torch.zeros((len(preview_feature_names),), dtype=torch.float32)
        preview_std = torch.ones((len(preview_feature_names),), dtype=torch.float32)
    preview_x_train_n = _normalize_inputs(preview_x_train, preview_mean.reshape(1, 1, -1), preview_std.reshape(1, 1, -1))
    preview_x_val_n = _normalize_inputs(preview_x_val, preview_mean.reshape(1, 1, -1), preview_std.reshape(1, 1, -1))

    state_value_mean = float(y_state_value_train.mean().item()) if y_state_value_train.numel() > 0 else 0.0
    state_value_std = float(y_state_value_train.std(unbiased=False).item()) if y_state_value_train.numel() > 0 else 1.0
    state_value_std = max(1.0e-6, state_value_std)
    y_state_value_train_n = (y_state_value_train - state_value_mean) / state_value_std
    y_state_value_val_n = (y_state_value_val - state_value_mean) / state_value_std

    model = _BuildTupleRankerNet(int(x.shape[1]), len(preview_feature_names), int(hidden_dim)).to(dtype=torch.float32)
    if isinstance(init_bundle, dict) and str(init_bundle.get("model_kind", "")) == REPAIR_CRITIC_BUILD_TUPLE_MODEL_KIND:
        _maybe_init_model_from_bundle(model, init_bundle)
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))

    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_val = float("inf")
    patience = max(20, int(0.20 * max(1, int(epochs))))
    bad_epochs = 0

    def _build_loss(
        outputs: dict[str, torch.Tensor],
        target_preview_utility: torch.Tensor,
        target_preview_mask: torch.Tensor,
        target_full_mask: torch.Tensor,
        target_pairs: Sequence[Sequence[tuple[int, int, float]]],
        target_state_value: torch.Tensor,
    ) -> torch.Tensor:
        preview_scores = outputs["preview_score"]
        loss = preview_scores.sum() * 0.0
        listwise_rows = target_full_mask & (target_preview_mask.sum(dim=-1) >= 2)
        if bool(listwise_rows.any().item()):
            loss = loss + _listwise_slate_loss(
                preview_scores[listwise_rows],
                target_preview_utility[listwise_rows],
                target_preview_mask[listwise_rows],
                beta=float(listwise_beta),
                min_gap=float(score_gap_floor),
            )
        partial_rows = (~target_full_mask).to(torch.bool)
        partial_scores = preview_scores[partial_rows]
        partial_pairs = [target_pairs[idx] for idx, keep in enumerate(partial_rows.tolist()) if bool(keep)]
        if partial_scores.shape[0] > 0:
            loss = loss + _pairwise_rank_loss(
                partial_scores,
                partial_pairs,
                gap_scale=float(pairwise_gap_scale),
            )
        if float(state_value_weight) > 0.0:
            loss = loss + float(state_value_weight) * F.smooth_l1_loss(
                outputs["state_value"],
                target_state_value,
            )
        return loss

    for _epoch in range(max(1, int(epochs))):
        model.train()
        outputs = model(x_train_n, preview_x_train_n)
        loss = _build_loss(
            outputs,
            y_preview_utility_train,
            preview_mask_train,
            full_slate_mask_train,
            pairwise_pairs_train,
            y_state_value_train_n,
        )
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        opt.step()

        model.eval()
        with torch.no_grad():
            outputs_val = model(x_val_n, preview_x_val_n)
            val_loss = _build_loss(
                outputs_val,
                y_preview_utility_val,
                preview_mask_val,
                full_slate_mask_val,
                pairwise_pairs_val,
                y_state_value_val_n,
            )
            val_loss_f = float(val_loss.item())
        if val_loss_f + 1.0e-8 < best_val:
            best_val = val_loss_f
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        train_out = model(x_train_n, preview_x_train_n)
        val_out = model(x_val_n, preview_x_val_n)
        train_metrics = _slate_rank_metrics(
            train_out["preview_score"],
            y_preview_utility_train,
            preview_mask_train,
            pairwise_pairs_train,
            prefix="build_tuple",
        )
        val_metrics = _slate_rank_metrics(
            val_out["preview_score"],
            y_preview_utility_val,
            preview_mask_val,
            pairwise_pairs_val,
            prefix="build_tuple",
        )
        train_metrics.update(_regression_metrics(
            train_out["state_value"] * state_value_std + state_value_mean,
            y_state_value_train,
            prefix="state_value",
        ))
        val_metrics.update(_regression_metrics(
            val_out["state_value"] * state_value_std + state_value_mean,
            y_state_value_val,
            prefix="state_value",
        ))

    return {
        "model_kind": REPAIR_CRITIC_BUILD_TUPLE_MODEL_KIND,
        "feature_names": feature_names,
        "build_preview_feature_names": preview_feature_names,
        "head_names": [],
        "hidden_dim": int(hidden_dim),
        "feature_mean": mean.detach().cpu(),
        "feature_std": std.detach().cpu(),
        "build_preview_feature_mean": preview_mean.detach().cpu(),
        "build_preview_feature_std": preview_std.detach().cpu(),
        "build_tuple_ranker_trained": True,
        "build_tuple_ranker_target": "same_parent_best_exact_utility",
        "build_tuple_ranker_score_gap_floor": float(max(0.0, score_gap_floor)),
        "build_tuple_ranker_state_value_weight": float(max(0.0, state_value_weight)),
        "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "metrics": {
            "n_examples": int(len(training_rows)),
            "n_full_slate_examples": int(full_slate_mask.sum().item()),
            "n_pairwise_examples": int((~full_slate_mask).sum().item()),
            "n_pairwise_pairs": int(sum(len(pairs) for pairs in pairwise_pairs)),
            "n_tuple_examples": int(preview_mask.sum().item()),
            "n_train": int(x_train.shape[0]),
            "n_val": int(x_val.shape[0]),
            "best_val_loss": float(best_val),
            "train": train_metrics,
            "val": val_metrics,
        },
    }


def train_repair_critic(
    rows: Sequence[Any],
    *,
    hidden_dim: int = 32,
    epochs: int = 250,
    lr: float = 1.0e-2,
    weight_decay: float = 1.0e-4,
    val_fraction: float = 0.2,
    seed: int = 0,
    utility_weights: dict[str, float] | None = None,
    init_bundle: dict[str, Any] | None = None,
) -> dict[str, Any]:
    torch.manual_seed(int(seed))
    training_rows, reward_scale = _build_training_rows(rows, utility_weights=utility_weights)
    aux_examples = [row for row in training_rows if isinstance(row.get("aux_target"), dict)]
    if len(aux_examples) < 8:
        raise ValueError("Need at least 8 supervised repair-critic examples to train a model.")

    feature_names = list(REPAIR_CRITIC_FEATURE_NAMES)
    head_names = list(REPAIR_CRITIC_HEAD_NAMES)
    macro_action_names = list(REPAIR_CRITIC_MACRO_ACTION_NAMES)
    route_names = list(REPAIR_CRITIC_ROUTE_NAMES)
    macro_index = {name: idx for idx, name in enumerate(macro_action_names)}
    route_index = {name: idx for idx, name in enumerate(route_names)}
    path_feature_names = list(REPAIR_CRITIC_PATH_FEATURE_NAMES)
    n_rows = len(training_rows)
    x = torch.tensor(
        [[float(row["features"].get(name, 0.0)) for name in feature_names] for row in training_rows],
        dtype=torch.float32,
    )
    y_aux = torch.zeros((n_rows, len(head_names)), dtype=torch.float32)
    aux_mask = torch.zeros((n_rows,), dtype=torch.bool)
    y_macro = torch.full((n_rows,), -100, dtype=torch.long)
    macro_mask = torch.zeros((n_rows,), dtype=torch.bool)
    y_route = torch.full((n_rows,), -100, dtype=torch.long)
    route_mask = torch.zeros((n_rows,), dtype=torch.bool)
    y_path = torch.full((n_rows,), -100, dtype=torch.long)
    path_label_mask = torch.zeros((n_rows,), dtype=torch.bool)

    max_paths = max((len(tuple(row.get("path_rows", ()))) for row in training_rows), default=0)
    max_paths = max(1, int(max_paths))
    path_x = torch.zeros((n_rows, max_paths, len(path_feature_names)), dtype=torch.float32)
    path_valid_mask = torch.zeros((n_rows, max_paths), dtype=torch.bool)

    for i, row in enumerate(training_rows):
        aux_target = row.get("aux_target")
        if isinstance(aux_target, dict):
            aux_mask[i] = True
            y_aux[i] = torch.tensor([float(aux_target[name]) for name in head_names], dtype=torch.float32)
        macro_name = row.get("macro_action", None)
        if isinstance(macro_name, str) and macro_name in macro_index:
            macro_mask[i] = True
            y_macro[i] = int(macro_index[macro_name])
        route_name = row.get("route_name", None)
        if isinstance(route_name, str) and route_name in route_index:
            route_mask[i] = True
            y_route[i] = int(route_index[route_name])
        path_rows = tuple(row.get("path_rows", ()) or ())
        for j, path_row in enumerate(path_rows[:max_paths]):
            path_valid_mask[i, j] = True
            path_x[i, j] = torch.tensor(
                repair_path_feature_vector(path_row, feature_names=path_feature_names),
                dtype=torch.float32,
            )
        path_target_index = row.get("path_target_index", None)
        if isinstance(path_target_index, int) and 0 <= path_target_index < min(len(path_rows), max_paths):
            path_label_mask[i] = True
            y_path[i] = int(path_target_index)

    train_idx, val_idx = _split_indices(n_rows, float(val_fraction), int(seed))
    x_train = x[train_idx]
    x_val = x[val_idx] if len(val_idx) > 0 else x_train
    y_aux_train = y_aux[train_idx]
    y_aux_val = y_aux[val_idx] if len(val_idx) > 0 else y_aux_train
    aux_mask_train = aux_mask[train_idx]
    aux_mask_val = aux_mask[val_idx] if len(val_idx) > 0 else aux_mask_train
    y_macro_train = y_macro[train_idx]
    y_macro_val = y_macro[val_idx] if len(val_idx) > 0 else y_macro_train
    macro_mask_train = macro_mask[train_idx]
    macro_mask_val = macro_mask[val_idx] if len(val_idx) > 0 else macro_mask_train
    y_route_train = y_route[train_idx]
    y_route_val = y_route[val_idx] if len(val_idx) > 0 else y_route_train
    route_mask_train = route_mask[train_idx]
    route_mask_val = route_mask[val_idx] if len(val_idx) > 0 else route_mask_train
    macro_route_mask_train = macro_mask_train & route_mask_train
    macro_route_mask_val = macro_mask_val & route_mask_val
    y_path_train = y_path[train_idx]
    y_path_val = y_path[val_idx] if len(val_idx) > 0 else y_path_train
    path_label_mask_train = path_label_mask[train_idx]
    path_label_mask_val = path_label_mask[val_idx] if len(val_idx) > 0 else path_label_mask_train
    tuple_route_mask_train = path_label_mask_train & route_mask_train
    tuple_route_mask_val = path_label_mask_val & route_mask_val
    tuple_macro_route_mask_train = path_label_mask_train & macro_route_mask_train
    tuple_macro_route_mask_val = path_label_mask_val & macro_route_mask_val
    path_x_train = path_x[train_idx]
    path_x_val = path_x[val_idx] if len(val_idx) > 0 else path_x_train
    path_valid_mask_train = path_valid_mask[train_idx]
    path_valid_mask_val = path_valid_mask[val_idx] if len(val_idx) > 0 else path_valid_mask_train

    mean, std = _compute_feature_stats(x_train)
    x_train_n = _normalize_inputs(x_train, mean, std)
    x_val_n = _normalize_inputs(x_val, mean, std)

    if bool(path_valid_mask_train.any().item()):
        path_train_flat = path_x_train[path_valid_mask_train]
        path_mean, path_std = _compute_feature_stats(path_train_flat)
    else:
        path_mean = torch.zeros((len(path_feature_names),), dtype=torch.float32)
        path_std = torch.ones((len(path_feature_names),), dtype=torch.float32)
    path_x_train_n = _normalize_inputs(path_x_train, path_mean.reshape(1, 1, -1), path_std.reshape(1, 1, -1))
    path_x_val_n = _normalize_inputs(path_x_val, path_mean.reshape(1, 1, -1), path_std.reshape(1, 1, -1))
    route_action_mask = _route_action_mask(macro_action_names, route_names)
    route_class_weights = _inverse_frequency_class_weights(y_route_train[route_mask_train], len(route_names))

    model = _RepairControllerSharedNet(
        int(x.shape[1]),
        int(hidden_dim),
        n_macro_actions=len(macro_action_names),
        n_routes=len(route_names),
        path_input_dim=len(path_feature_names),
    ).to(dtype=torch.float32)
    _maybe_init_model_from_bundle(model, init_bundle)
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))

    if bool(aux_mask_train.any().item()):
        aux_y_train_labeled = y_aux_train[aux_mask_train]
    else:
        aux_y_train_labeled = y_aux_train[:0]
    pos_weights = [
        _binary_head_pos_weight(aux_y_train_labeled[:, i]) if aux_y_train_labeled.shape[0] > 0 else torch.tensor(1.0)
        for i in range(4)
    ]

    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_val = float("inf")
    patience = max(20, int(0.15 * max(1, int(epochs))))
    bad_epochs = 0

    for _epoch in range(max(1, int(epochs))):
        model.train()
        outputs = model(x_train_n, path_x_train_n, path_valid_mask_train)
        loss = outputs["aux_logits"].sum() * 0.0
        if bool(aux_mask_train.any().item()):
            loss = loss + _aux_loss_from_logits(
                outputs["aux_logits"][aux_mask_train],
                y_aux_train[aux_mask_train],
                pos_weights=pos_weights,
            )
        if bool(route_mask_train.any().item()):
            loss = loss + F.cross_entropy(
                outputs["route_logits"][route_mask_train],
                y_route_train[route_mask_train],
                weight=route_class_weights,
            )
        if bool(tuple_route_mask_train.any().item()):
            tuple_route_logits = _gather_path_head(
                outputs["path_route_logits"][tuple_route_mask_train],
                y_path_train[tuple_route_mask_train],
            )
            loss = loss + F.cross_entropy(
                tuple_route_logits,
                y_route_train[tuple_route_mask_train],
                weight=route_class_weights,
            )
        if bool(macro_route_mask_train.any().item()):
            loss = loss + F.cross_entropy(
                _route_masked_action_logits(
                    outputs["macro_logits"][macro_route_mask_train],
                    y_route_train[macro_route_mask_train],
                    route_action_mask,
                ),
                y_macro_train[macro_route_mask_train],
            )
        elif bool(macro_mask_train.any().item()):
            loss = loss + F.cross_entropy(outputs["macro_logits"][macro_mask_train], y_macro_train[macro_mask_train])
        if bool(tuple_macro_route_mask_train.any().item()):
            tuple_macro_logits = _gather_path_head(
                outputs["path_macro_logits"][tuple_macro_route_mask_train],
                y_path_train[tuple_macro_route_mask_train],
            )
            loss = loss + F.cross_entropy(
                _route_masked_action_logits(
                    tuple_macro_logits,
                    y_route_train[tuple_macro_route_mask_train],
                    route_action_mask,
                ),
                y_macro_train[tuple_macro_route_mask_train],
            )
        if bool(path_label_mask_train.any().item()):
            loss = loss + _masked_path_cross_entropy(
                outputs["path_logits"][path_label_mask_train],
                y_path_train[path_label_mask_train],
                path_valid_mask_train[path_label_mask_train],
            )

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        opt.step()

        model.eval()
        with torch.no_grad():
            outputs_val = model(x_val_n, path_x_val_n, path_valid_mask_val)
            val_loss = outputs_val["aux_logits"].sum() * 0.0
            if bool(aux_mask_val.any().item()):
                val_loss = val_loss + _aux_loss_from_logits(
                    outputs_val["aux_logits"][aux_mask_val],
                    y_aux_val[aux_mask_val],
                    pos_weights=pos_weights,
                )
            if bool(route_mask_val.any().item()):
                val_loss = val_loss + F.cross_entropy(
                    outputs_val["route_logits"][route_mask_val],
                    y_route_val[route_mask_val],
                    weight=route_class_weights,
                )
            if bool(tuple_route_mask_val.any().item()):
                tuple_route_logits_val = _gather_path_head(
                    outputs_val["path_route_logits"][tuple_route_mask_val],
                    y_path_val[tuple_route_mask_val],
                )
                val_loss = val_loss + F.cross_entropy(
                    tuple_route_logits_val,
                    y_route_val[tuple_route_mask_val],
                    weight=route_class_weights,
                )
            if bool(macro_route_mask_val.any().item()):
                val_loss = val_loss + F.cross_entropy(
                    _route_masked_action_logits(
                        outputs_val["macro_logits"][macro_route_mask_val],
                        y_route_val[macro_route_mask_val],
                        route_action_mask,
                    ),
                    y_macro_val[macro_route_mask_val],
                )
            elif bool(macro_mask_val.any().item()):
                val_loss = val_loss + F.cross_entropy(outputs_val["macro_logits"][macro_mask_val], y_macro_val[macro_mask_val])
            if bool(tuple_macro_route_mask_val.any().item()):
                tuple_macro_logits_val = _gather_path_head(
                    outputs_val["path_macro_logits"][tuple_macro_route_mask_val],
                    y_path_val[tuple_macro_route_mask_val],
                )
                val_loss = val_loss + F.cross_entropy(
                    _route_masked_action_logits(
                        tuple_macro_logits_val,
                        y_route_val[tuple_macro_route_mask_val],
                        route_action_mask,
                    ),
                    y_macro_val[tuple_macro_route_mask_val],
                )
            if bool(path_label_mask_val.any().item()):
                val_loss = val_loss + _masked_path_cross_entropy(
                    outputs_val["path_logits"][path_label_mask_val],
                    y_path_val[path_label_mask_val],
                    path_valid_mask_val[path_label_mask_val],
                )
            val_loss_f = float(val_loss.item())
        if val_loss_f + 1.0e-8 < best_val:
            best_val = val_loss_f
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        train_out = model(x_train_n, path_x_train_n, path_valid_mask_train)
        val_out = model(x_val_n, path_x_val_n, path_valid_mask_val)
        train_metrics: dict[str, float] = {}
        val_metrics: dict[str, float] = {}
        if bool(aux_mask_train.any().item()):
            train_aux_preds = _aux_predictions_from_logits(train_out["aux_logits"][aux_mask_train], head_names=head_names)
            train_metrics.update(_metrics_from_preds(train_aux_preds, y_aux_train[aux_mask_train], head_names=head_names))
        if bool(aux_mask_val.any().item()):
            val_aux_preds = _aux_predictions_from_logits(val_out["aux_logits"][aux_mask_val], head_names=head_names)
            val_metrics.update(_metrics_from_preds(val_aux_preds, y_aux_val[aux_mask_val], head_names=head_names))
        if bool(macro_mask_train.any().item()):
            train_metrics.update(_macro_metrics_from_logits(
                train_out["macro_logits"][macro_mask_train],
                y_macro_train[macro_mask_train],
                action_names=macro_action_names,
            ))
        if bool(macro_route_mask_train.any().item()):
            train_metrics.update(_hierarchical_macro_metrics(
                train_out["route_logits"][macro_route_mask_train],
                train_out["macro_logits"][macro_route_mask_train],
                y_route_train[macro_route_mask_train],
                y_macro_train[macro_route_mask_train],
                route_action_mask,
            ))
        if bool(tuple_macro_route_mask_train.any().item()):
            train_tuple_route = _gather_path_head(
                train_out["path_route_logits"][tuple_macro_route_mask_train],
                y_path_train[tuple_macro_route_mask_train],
            )
            train_tuple_macro = _gather_path_head(
                train_out["path_macro_logits"][tuple_macro_route_mask_train],
                y_path_train[tuple_macro_route_mask_train],
            )
            train_metrics.update(_hierarchical_macro_metrics(
                train_tuple_route,
                train_tuple_macro,
                y_route_train[tuple_macro_route_mask_train],
                y_macro_train[tuple_macro_route_mask_train],
                route_action_mask,
            ))
        if bool(route_mask_train.any().item()):
            train_metrics.update(_flat_classification_metrics(
                train_out["route_logits"][route_mask_train],
                y_route_train[route_mask_train],
                prefix="route",
            ))
        if bool(tuple_route_mask_train.any().item()):
            train_metrics.update(_flat_classification_metrics(
                _gather_path_head(
                    train_out["path_route_logits"][tuple_route_mask_train],
                    y_path_train[tuple_route_mask_train],
                ),
                y_route_train[tuple_route_mask_train],
                prefix="tuple_route",
            ))
        if bool(macro_mask_val.any().item()):
            val_metrics.update(_macro_metrics_from_logits(
                val_out["macro_logits"][macro_mask_val],
                y_macro_val[macro_mask_val],
                action_names=macro_action_names,
            ))
        if bool(macro_route_mask_val.any().item()):
            val_metrics.update(_hierarchical_macro_metrics(
                val_out["route_logits"][macro_route_mask_val],
                val_out["macro_logits"][macro_route_mask_val],
                y_route_val[macro_route_mask_val],
                y_macro_val[macro_route_mask_val],
                route_action_mask,
            ))
        if bool(tuple_macro_route_mask_val.any().item()):
            val_tuple_route = _gather_path_head(
                val_out["path_route_logits"][tuple_macro_route_mask_val],
                y_path_val[tuple_macro_route_mask_val],
            )
            val_tuple_macro = _gather_path_head(
                val_out["path_macro_logits"][tuple_macro_route_mask_val],
                y_path_val[tuple_macro_route_mask_val],
            )
            val_metrics.update(_hierarchical_macro_metrics(
                val_tuple_route,
                val_tuple_macro,
                y_route_val[tuple_macro_route_mask_val],
                y_macro_val[tuple_macro_route_mask_val],
                route_action_mask,
            ))
        if bool(route_mask_val.any().item()):
            val_metrics.update(_flat_classification_metrics(
                val_out["route_logits"][route_mask_val],
                y_route_val[route_mask_val],
                prefix="route",
            ))
        if bool(tuple_route_mask_val.any().item()):
            val_metrics.update(_flat_classification_metrics(
                _gather_path_head(
                    val_out["path_route_logits"][tuple_route_mask_val],
                    y_path_val[tuple_route_mask_val],
                ),
                y_route_val[tuple_route_mask_val],
                prefix="tuple_route",
            ))
        if bool(path_label_mask_train.any().item()):
            train_metrics.update(_path_metrics_from_logits(
                train_out["path_logits"][path_label_mask_train],
                y_path_train[path_label_mask_train],
                path_valid_mask_train[path_label_mask_train],
            ))
        if bool(path_label_mask_val.any().item()):
            val_metrics.update(_path_metrics_from_logits(
                val_out["path_logits"][path_label_mask_val],
                y_path_val[path_label_mask_val],
                path_valid_mask_val[path_label_mask_val],
            ))

    weights_out = dict(REPAIR_CRITIC_DEFAULT_UTILITY_WEIGHTS)
    if isinstance(utility_weights, dict):
        for key, value in utility_weights.items():
            if key in weights_out:
                weights_out[key] = max(0.0, _to_float(value, weights_out[key]))
    return {
        "model_kind": REPAIR_CRITIC_SHARED_MODEL_KIND,
        "feature_names": feature_names,
        "path_feature_names": path_feature_names,
        "head_names": head_names,
        "macro_action_names": macro_action_names,
        "route_names": route_names,
        "mode_names": list(REPAIR_CRITIC_MODE_NAMES),
        "path_relation_names": list(REPAIR_CRITIC_PATH_RELATION_NAMES),
        "hidden_dim": int(hidden_dim),
        "reward_per_s_scale": float(reward_scale),
        "actor_critic_reward_target": str(init_bundle.get("actor_critic_reward_target", "immediate")) if isinstance(init_bundle, dict) else "immediate",
        "actor_critic_reward_mean": float(init_bundle.get("actor_critic_reward_mean", 0.0)) if isinstance(init_bundle, dict) else 0.0,
        "actor_critic_reward_std": float(init_bundle.get("actor_critic_reward_std", 1.0)) if isinstance(init_bundle, dict) else 1.0,
        "feature_mean": mean.detach().cpu(),
        "feature_std": std.detach().cpu(),
        "path_feature_mean": path_mean.detach().cpu(),
        "path_feature_std": path_std.detach().cpu(),
        "utility_weights": weights_out,
        "macro_head_trained": bool(macro_mask.any().item()),
        "route_head_trained": bool(route_mask.any().item()),
        "q_head_trained": bool(init_bundle.get("q_head_trained", False)) if isinstance(init_bundle, dict) else False,
        "value_head_trained": bool(init_bundle.get("value_head_trained", False)) if isinstance(init_bundle, dict) else False,
        "path_head_trained": bool(path_label_mask.any().item()),
        "path_action_head_trained": bool(tuple_macro_route_mask_train.any().item() or tuple_macro_route_mask_val.any().item()),
        "path_relation_head_trained": bool(init_bundle.get("path_relation_head_trained", False)) if isinstance(init_bundle, dict) else False,
        "path_mode_head_trained": bool(init_bundle.get("path_mode_head_trained", False)) if isinstance(init_bundle, dict) else False,
        "path_improve_head_trained": bool(init_bundle.get("path_improve_head_trained", False)) if isinstance(init_bundle, dict) else False,
        "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "metrics": {
            "n_examples": int(len(training_rows)),
            "n_aux_examples": int(aux_mask.sum().item()),
            "n_macro_examples": int(macro_mask.sum().item()),
            "n_route_examples": int(route_mask.sum().item()),
            "n_path_examples": int(path_label_mask.sum().item()),
            "n_train": int(x_train.shape[0]),
            "n_val": int(x_val.shape[0]),
            "best_val_loss": float(best_val),
            "train": train_metrics,
            "val": val_metrics,
        },
    }


def pretrain_repair_controller_from_oracle_tasks(
    tasks: Sequence[Any],
    *,
    hidden_dim: int = 32,
    epochs: int = 200,
    lr: float = 1.0e-2,
    weight_decay: float = 1.0e-4,
    val_fraction: float = 0.2,
    seed: int = 0,
    init_bundle: dict[str, Any] | None = None,
) -> dict[str, Any]:
    torch.manual_seed(int(seed))
    training_rows = _build_oracle_pretrain_rows(tasks)
    if len(training_rows) < 4:
        raise ValueError("Need at least 4 oracle pretraining rows to train controller heads.")

    feature_names = list(REPAIR_CRITIC_FEATURE_NAMES)
    path_feature_names = list(REPAIR_CRITIC_PATH_FEATURE_NAMES)
    macro_action_names = list(REPAIR_CRITIC_MACRO_ACTION_NAMES)
    route_names = list(REPAIR_CRITIC_ROUTE_NAMES)
    mode_names = list(REPAIR_CRITIC_MODE_NAMES)
    relation_names = list(REPAIR_CRITIC_PATH_RELATION_NAMES)
    n_rows = len(training_rows)
    max_paths = max((len(tuple(row.get("path_rows", ()))) for row in training_rows), default=0)
    max_paths = max(1, int(max_paths))

    x = torch.tensor(
        [[float(row["features"].get(name, 0.0)) for name in feature_names] for row in training_rows],
        dtype=torch.float32,
    )
    path_x = torch.zeros((n_rows, max_paths, len(path_feature_names)), dtype=torch.float32)
    path_valid_mask = torch.zeros((n_rows, max_paths), dtype=torch.bool)
    y_path = torch.full((n_rows,), -100, dtype=torch.long)
    path_label_mask = torch.zeros((n_rows,), dtype=torch.bool)
    y_relation = torch.full((n_rows, max_paths), -100, dtype=torch.long)
    relation_mask = torch.zeros((n_rows, max_paths), dtype=torch.bool)
    y_mode = torch.full((n_rows, max_paths), -100, dtype=torch.long)
    mode_mask = torch.zeros((n_rows, max_paths), dtype=torch.bool)
    y_improve = torch.zeros((n_rows, max_paths), dtype=torch.float32)
    improve_mask = torch.zeros((n_rows, max_paths), dtype=torch.bool)

    for i, row in enumerate(training_rows):
        path_rows = tuple(row.get("path_rows", ()) or ())
        for j, path_row in enumerate(path_rows[:max_paths]):
            path_valid_mask[i, j] = True
            path_x[i, j] = torch.tensor(
                repair_path_feature_vector(path_row, feature_names=path_feature_names),
                dtype=torch.float32,
            )
        path_target_index = row.get("path_target_index", None)
        if isinstance(path_target_index, int) and 0 <= path_target_index < min(len(path_rows), max_paths):
            path_label_mask[i] = True
            y_path[i] = int(path_target_index)
        for j, target in enumerate(list(row.get("relation_targets", []) or [])[:max_paths]):
            if isinstance(target, int) and target >= 0:
                relation_mask[i, j] = True
                y_relation[i, j] = int(target)
        for j, target in enumerate(list(row.get("mode_targets", []) or [])[:max_paths]):
            if isinstance(target, int) and target >= 0:
                mode_mask[i, j] = True
                y_mode[i, j] = int(target)
        improve_targets = list(row.get("improve_targets", []) or [])
        improve_flags = list(row.get("improve_mask", []) or [])
        for j in range(min(max_paths, len(improve_targets), len(improve_flags))):
            if bool(improve_flags[j]):
                improve_mask[i, j] = True
                y_improve[i, j] = float(improve_targets[j])

    train_idx, val_idx = _split_indices(n_rows, float(val_fraction), int(seed))
    x_train = x[train_idx]
    x_val = x[val_idx] if len(val_idx) > 0 else x_train
    path_x_train = path_x[train_idx]
    path_x_val = path_x[val_idx] if len(val_idx) > 0 else path_x_train
    path_valid_mask_train = path_valid_mask[train_idx]
    path_valid_mask_val = path_valid_mask[val_idx] if len(val_idx) > 0 else path_valid_mask_train
    y_path_train = y_path[train_idx]
    y_path_val = y_path[val_idx] if len(val_idx) > 0 else y_path_train
    path_label_mask_train = path_label_mask[train_idx]
    path_label_mask_val = path_label_mask[val_idx] if len(val_idx) > 0 else path_label_mask_train
    y_relation_train = y_relation[train_idx]
    y_relation_val = y_relation[val_idx] if len(val_idx) > 0 else y_relation_train
    relation_mask_train = relation_mask[train_idx]
    relation_mask_val = relation_mask[val_idx] if len(val_idx) > 0 else relation_mask_train
    y_mode_train = y_mode[train_idx]
    y_mode_val = y_mode[val_idx] if len(val_idx) > 0 else y_mode_train
    mode_mask_train = mode_mask[train_idx]
    mode_mask_val = mode_mask[val_idx] if len(val_idx) > 0 else mode_mask_train
    y_improve_train = y_improve[train_idx]
    y_improve_val = y_improve[val_idx] if len(val_idx) > 0 else y_improve_train
    improve_mask_train = improve_mask[train_idx]
    improve_mask_val = improve_mask[val_idx] if len(val_idx) > 0 else improve_mask_train

    mean, std = _compute_feature_stats(x_train)
    x_train_n = _normalize_inputs(x_train, mean, std)
    x_val_n = _normalize_inputs(x_val, mean, std)
    if bool(path_valid_mask_train.any().item()):
        path_train_flat = path_x_train[path_valid_mask_train]
        path_mean, path_std = _compute_feature_stats(path_train_flat)
    else:
        path_mean = torch.zeros((len(path_feature_names),), dtype=torch.float32)
        path_std = torch.ones((len(path_feature_names),), dtype=torch.float32)
    path_x_train_n = _normalize_inputs(path_x_train, path_mean.reshape(1, 1, -1), path_std.reshape(1, 1, -1))
    path_x_val_n = _normalize_inputs(path_x_val, path_mean.reshape(1, 1, -1), path_std.reshape(1, 1, -1))

    model = _RepairControllerSharedNet(
        int(x.shape[1]),
        int(hidden_dim),
        n_macro_actions=len(macro_action_names),
        n_routes=len(route_names),
        path_input_dim=len(path_feature_names),
    ).to(dtype=torch.float32)
    _maybe_init_model_from_bundle(model, init_bundle)
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))

    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_val = float("inf")
    patience = max(20, int(0.15 * max(1, int(epochs))))
    bad_epochs = 0

    for _epoch in range(max(1, int(epochs))):
        model.train()
        outputs = model(x_train_n, path_x_train_n, path_valid_mask_train)
        loss = outputs["path_logits"].sum() * 0.0
        if bool(path_label_mask_train.any().item()):
            loss = loss + _masked_path_cross_entropy(
                outputs["path_logits"][path_label_mask_train],
                y_path_train[path_label_mask_train],
                path_valid_mask_train[path_label_mask_train],
            )
        if bool(relation_mask_train.any().item()):
            rel_logits = outputs["path_relation_logits"][relation_mask_train]
            loss = loss + F.cross_entropy(rel_logits, y_relation_train[relation_mask_train])
        if bool(mode_mask_train.any().item()):
            mode_logits = outputs["path_mode_logits"][mode_mask_train]
            loss = loss + F.cross_entropy(mode_logits, y_mode_train[mode_mask_train])
        if bool(improve_mask_train.any().item()):
            improve_pred = torch.sigmoid(outputs["path_improve"][improve_mask_train])
            loss = loss + F.smooth_l1_loss(improve_pred, y_improve_train[improve_mask_train])

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        opt.step()

        model.eval()
        with torch.no_grad():
            outputs_val = model(x_val_n, path_x_val_n, path_valid_mask_val)
            val_loss = outputs_val["path_logits"].sum() * 0.0
            if bool(path_label_mask_val.any().item()):
                val_loss = val_loss + _masked_path_cross_entropy(
                    outputs_val["path_logits"][path_label_mask_val],
                    y_path_val[path_label_mask_val],
                    path_valid_mask_val[path_label_mask_val],
                )
            if bool(relation_mask_val.any().item()):
                val_loss = val_loss + F.cross_entropy(
                    outputs_val["path_relation_logits"][relation_mask_val],
                    y_relation_val[relation_mask_val],
                )
            if bool(mode_mask_val.any().item()):
                val_loss = val_loss + F.cross_entropy(
                    outputs_val["path_mode_logits"][mode_mask_val],
                    y_mode_val[mode_mask_val],
                )
            if bool(improve_mask_val.any().item()):
                improve_pred = torch.sigmoid(outputs_val["path_improve"][improve_mask_val])
                val_loss = val_loss + F.smooth_l1_loss(improve_pred, y_improve_val[improve_mask_val])
            val_loss_f = float(val_loss.item())
        if val_loss_f + 1.0e-8 < best_val:
            best_val = val_loss_f
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        train_out = model(x_train_n, path_x_train_n, path_valid_mask_train)
        val_out = model(x_val_n, path_x_val_n, path_valid_mask_val)
        train_metrics: dict[str, float] = {}
        val_metrics: dict[str, float] = {}
        if bool(path_label_mask_train.any().item()):
            train_metrics.update(_path_metrics_from_logits(
                train_out["path_logits"][path_label_mask_train],
                y_path_train[path_label_mask_train],
                path_valid_mask_train[path_label_mask_train],
            ))
        if bool(path_label_mask_val.any().item()):
            val_metrics.update(_path_metrics_from_logits(
                val_out["path_logits"][path_label_mask_val],
                y_path_val[path_label_mask_val],
                path_valid_mask_val[path_label_mask_val],
            ))
        if bool(relation_mask_train.any().item()):
            train_metrics.update(_flat_classification_metrics(
                train_out["path_relation_logits"][relation_mask_train],
                y_relation_train[relation_mask_train],
                prefix="path_relation",
            ))
        if bool(relation_mask_val.any().item()):
            val_metrics.update(_flat_classification_metrics(
                val_out["path_relation_logits"][relation_mask_val],
                y_relation_val[relation_mask_val],
                prefix="path_relation",
            ))
        if bool(mode_mask_train.any().item()):
            train_metrics.update(_flat_classification_metrics(
                train_out["path_mode_logits"][mode_mask_train],
                y_mode_train[mode_mask_train],
                prefix="path_mode",
            ))
        if bool(mode_mask_val.any().item()):
            val_metrics.update(_flat_classification_metrics(
                val_out["path_mode_logits"][mode_mask_val],
                y_mode_val[mode_mask_val],
                prefix="path_mode",
            ))
        if bool(improve_mask_train.any().item()):
            train_metrics.update(_regression_metrics(
                torch.sigmoid(train_out["path_improve"][improve_mask_train]),
                y_improve_train[improve_mask_train],
                prefix="path_improve",
            ))
        if bool(improve_mask_val.any().item()):
            val_metrics.update(_regression_metrics(
                torch.sigmoid(val_out["path_improve"][improve_mask_val]),
                y_improve_val[improve_mask_val],
                prefix="path_improve",
            ))

    utility_weights = dict(REPAIR_CRITIC_DEFAULT_UTILITY_WEIGHTS)
    if isinstance(init_bundle, dict) and isinstance(init_bundle.get("utility_weights"), dict):
        utility_weights.update(init_bundle["utility_weights"])
    return {
        "model_kind": REPAIR_CRITIC_SHARED_MODEL_KIND,
        "feature_names": feature_names,
        "path_feature_names": path_feature_names,
        "head_names": list(REPAIR_CRITIC_HEAD_NAMES),
        "macro_action_names": macro_action_names,
        "route_names": route_names,
        "mode_names": mode_names,
        "path_relation_names": relation_names,
        "hidden_dim": int(hidden_dim),
        "reward_per_s_scale": float(init_bundle.get("reward_per_s_scale", 1.0)) if isinstance(init_bundle, dict) else 1.0,
        "actor_critic_reward_target": str(init_bundle.get("actor_critic_reward_target", "immediate")) if isinstance(init_bundle, dict) else "immediate",
        "actor_critic_reward_mean": float(init_bundle.get("actor_critic_reward_mean", 0.0)) if isinstance(init_bundle, dict) else 0.0,
        "actor_critic_reward_std": float(init_bundle.get("actor_critic_reward_std", 1.0)) if isinstance(init_bundle, dict) else 1.0,
        "feature_mean": mean.detach().cpu(),
        "feature_std": std.detach().cpu(),
        "path_feature_mean": path_mean.detach().cpu(),
        "path_feature_std": path_std.detach().cpu(),
        "utility_weights": utility_weights,
        "macro_head_trained": bool(init_bundle.get("macro_head_trained", False)) if isinstance(init_bundle, dict) else False,
        "route_head_trained": bool(init_bundle.get("route_head_trained", False)) if isinstance(init_bundle, dict) else False,
        "q_head_trained": bool(init_bundle.get("q_head_trained", False)) if isinstance(init_bundle, dict) else False,
        "value_head_trained": bool(init_bundle.get("value_head_trained", False)) if isinstance(init_bundle, dict) else False,
        "path_head_trained": bool(path_label_mask.any().item()),
        "path_relation_head_trained": bool(relation_mask.any().item()),
        "path_mode_head_trained": bool(mode_mask.any().item()),
        "path_improve_head_trained": bool(improve_mask.any().item()),
        "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "metrics": {
            "n_examples": int(len(training_rows)),
            "n_path_examples": int(path_label_mask.sum().item()),
            "n_relation_examples": int(relation_mask.sum().item()),
            "n_mode_examples": int(mode_mask.sum().item()),
            "n_improve_examples": int(improve_mask.sum().item()),
            "n_train": int(x_train.shape[0]),
            "n_val": int(x_val.shape[0]),
            "best_val_loss": float(best_val),
            "train": train_metrics,
            "val": val_metrics,
        },
    }


def train_repair_controller_actor_critic(
    rows: Sequence[Any],
    *,
    hidden_dim: int = 32,
    epochs: int = 120,
    lr: float = 5.0e-3,
    weight_decay: float = 1.0e-4,
    val_fraction: float = 0.2,
    seed: int = 0,
    entropy_weight: float = 0.01,
    value_weight: float = 0.5,
    policy_ce_weight: float = 0.10,
    path_ce_weight: float = 0.10,
    advantage_clip: float = 5.0,
    reward_target: str = "descendant_preferred",
    init_bundle: dict[str, Any] | None = None,
) -> dict[str, Any]:
    torch.manual_seed(int(seed))
    training_rows = _build_actor_critic_rows(rows, reward_target=reward_target)
    if len(training_rows) < 8:
        raise ValueError("Need at least 8 actor-critic transition rows to train a controller policy.")

    feature_names = list(REPAIR_CRITIC_FEATURE_NAMES)
    path_feature_names = list(REPAIR_CRITIC_PATH_FEATURE_NAMES)
    head_names = list(REPAIR_CRITIC_HEAD_NAMES)
    macro_action_names = list(REPAIR_CRITIC_MACRO_ACTION_NAMES)
    route_names = list(REPAIR_CRITIC_ROUTE_NAMES)
    mode_names = list(REPAIR_CRITIC_MODE_NAMES)
    relation_names = list(REPAIR_CRITIC_PATH_RELATION_NAMES)
    macro_index = {name: idx for idx, name in enumerate(macro_action_names)}
    route_index = {name: idx for idx, name in enumerate(route_names)}

    n_rows = len(training_rows)
    x = torch.tensor(
        [[float(row["features"].get(name, 0.0)) for name in feature_names] for row in training_rows],
        dtype=torch.float32,
    )
    y_macro = torch.tensor(
        [int(macro_index[row["macro_action"]]) for row in training_rows],
        dtype=torch.long,
    )
    y_route = torch.tensor(
        [int(route_index.get(str(row.get("route_name", "") or ""), 0)) for row in training_rows],
        dtype=torch.long,
    )
    y_reward = torch.tensor(
        [float(row["actor_critic_reward"]) for row in training_rows],
        dtype=torch.float32,
    )
    reward_source_counts: dict[str, int] = {}
    for row in training_rows:
        reward_source = str(row.get("actor_critic_reward_source", "") or "unknown")
        reward_source_counts[reward_source] = int(reward_source_counts.get(reward_source, 0)) + 1
    max_paths = max((len(tuple(row.get("path_rows", ()))) for row in training_rows), default=0)
    max_paths = max(1, int(max_paths))
    path_x = torch.zeros((n_rows, max_paths, len(path_feature_names)), dtype=torch.float32)
    path_valid_mask = torch.zeros((n_rows, max_paths), dtype=torch.bool)
    y_path = torch.full((n_rows,), -100, dtype=torch.long)
    path_label_mask = torch.zeros((n_rows,), dtype=torch.bool)
    for i, row in enumerate(training_rows):
        path_rows = tuple(row.get("path_rows", ()) or ())
        for j, path_row in enumerate(path_rows[:max_paths]):
            path_valid_mask[i, j] = True
            path_x[i, j] = torch.tensor(
                repair_path_feature_vector(path_row, feature_names=path_feature_names),
                dtype=torch.float32,
            )
        path_target_index = row.get("path_target_index", None)
        if isinstance(path_target_index, int) and 0 <= path_target_index < min(len(path_rows), max_paths):
            y_path[i] = int(path_target_index)
            path_label_mask[i] = True

    train_idx, val_idx = _split_indices(n_rows, float(val_fraction), int(seed))
    x_train = x[train_idx]
    x_val = x[val_idx] if len(val_idx) > 0 else x_train
    path_x_train = path_x[train_idx]
    path_x_val = path_x[val_idx] if len(val_idx) > 0 else path_x_train
    path_valid_mask_train = path_valid_mask[train_idx]
    path_valid_mask_val = path_valid_mask[val_idx] if len(val_idx) > 0 else path_valid_mask_train
    y_path_train = y_path[train_idx]
    y_path_val = y_path[val_idx] if len(val_idx) > 0 else y_path_train
    path_label_mask_train = path_label_mask[train_idx]
    path_label_mask_val = path_label_mask[val_idx] if len(val_idx) > 0 else path_label_mask_train
    y_macro_train = y_macro[train_idx]
    y_macro_val = y_macro[val_idx] if len(val_idx) > 0 else y_macro_train
    y_route_train = y_route[train_idx]
    y_route_val = y_route[val_idx] if len(val_idx) > 0 else y_route_train
    y_reward_train = y_reward[train_idx]
    y_reward_val = y_reward[val_idx] if len(val_idx) > 0 else y_reward_train

    mean, std = _compute_feature_stats(x_train)
    x_train_n = _normalize_inputs(x_train, mean, std)
    x_val_n = _normalize_inputs(x_val, mean, std)
    if bool(path_valid_mask_train.any().item()):
        path_train_flat = path_x_train[path_valid_mask_train]
        path_mean, path_std = _compute_feature_stats(path_train_flat)
    elif isinstance(init_bundle, dict):
        path_mean = torch.as_tensor(
            init_bundle.get("path_feature_mean", torch.zeros(len(path_feature_names))),
            dtype=torch.float32,
        )
        path_std = torch.as_tensor(
            init_bundle.get("path_feature_std", torch.ones(len(path_feature_names))),
            dtype=torch.float32,
        )
    else:
        path_mean = torch.zeros((len(path_feature_names),), dtype=torch.float32)
        path_std = torch.ones((len(path_feature_names),), dtype=torch.float32)
    path_x_train_n = _normalize_inputs(path_x_train, path_mean.reshape(1, 1, -1), path_std.reshape(1, 1, -1))
    path_x_val_n = _normalize_inputs(path_x_val, path_mean.reshape(1, 1, -1), path_std.reshape(1, 1, -1))

    reward_mean = float(y_reward_train.mean().item()) if y_reward_train.numel() > 0 else 0.0
    reward_std = float(y_reward_train.std(unbiased=False).item()) if y_reward_train.numel() > 0 else 1.0
    reward_std = max(1.0e-6, reward_std)
    y_reward_train_n = (y_reward_train - reward_mean) / reward_std
    y_reward_val_n = (y_reward_val - reward_mean) / reward_std
    route_action_mask = _route_action_mask(macro_action_names, route_names)
    route_class_weights = _inverse_frequency_class_weights(y_route_train, len(route_names))
    tuple_route_mask_train = path_label_mask_train.clone()
    tuple_route_mask_val = path_label_mask_val.clone()

    model = _RepairControllerSharedNet(
        int(x.shape[1]),
        int(hidden_dim),
        n_macro_actions=len(macro_action_names),
        n_routes=len(route_names),
        path_input_dim=len(path_feature_names),
    ).to(dtype=torch.float32)
    _maybe_init_model_from_bundle(model, init_bundle)
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))

    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_val = float("inf")
    patience = max(20, int(0.20 * max(1, int(epochs))))
    bad_epochs = 0

    def _actor_critic_loss(
        outputs: dict[str, torch.Tensor],
        y_route_local: torch.Tensor,
        y_action: torch.Tensor,
        y_ret: torch.Tensor,
        y_path_local: torch.Tensor,
        path_label_mask_local: torch.Tensor,
        path_valid_mask_local: torch.Tensor,
    ) -> torch.Tensor:
        logits = outputs["macro_logits"]
        route_logits = outputs["route_logits"]
        q_values = outputs["q_values"]
        value_pred = outputs["value_pred"]
        tuple_mask = path_label_mask_local & (y_path_local >= 0)
        if bool(tuple_mask.any().item()):
            route_logits = route_logits.clone()
            logits = logits.clone()
            q_values = q_values.clone()
            tuple_route_logits = _gather_path_head(outputs["path_route_logits"][tuple_mask], y_path_local[tuple_mask])
            tuple_macro_logits = _gather_path_head(outputs["path_macro_logits"][tuple_mask], y_path_local[tuple_mask])
            tuple_q_values = _gather_path_head(outputs["path_q_values"][tuple_mask], y_path_local[tuple_mask])
            route_logits[tuple_mask] = tuple_route_logits
            logits[tuple_mask] = tuple_macro_logits
            q_values[tuple_mask] = tuple_q_values
        masked_logits = _route_masked_action_logits(logits, y_route_local, route_action_mask)
        route_log_probs = torch.log_softmax(route_logits, dim=-1)
        route_probs = torch.softmax(route_logits, dim=-1)
        within_route_log_probs = torch.log_softmax(masked_logits, dim=-1)
        within_route_probs = torch.softmax(masked_logits, dim=-1)
        chosen_route_log_prob = route_log_probs.gather(1, y_route_local.reshape(-1, 1)).reshape(-1)
        chosen_action_log_prob = within_route_log_probs.gather(1, y_action.reshape(-1, 1)).reshape(-1)
        chosen_log_prob = chosen_route_log_prob + chosen_action_log_prob
        chosen_q = q_values.gather(1, y_action.reshape(-1, 1)).reshape(-1)
        advantage = (y_ret - value_pred).detach()
        if float(advantage_clip) > 0.0:
            advantage = torch.clamp(advantage, min=-float(advantage_clip), max=float(advantage_clip))
        policy_loss = -(chosen_log_prob * advantage).mean()
        route_loss = F.cross_entropy(route_logits, y_route_local, weight=route_class_weights)
        q_loss = F.smooth_l1_loss(chosen_q, y_ret)
        value_loss = F.smooth_l1_loss(value_pred, y_ret)
        route_entropy = -(route_probs * route_log_probs).sum(dim=-1).mean()
        within_route_entropy = -(within_route_probs * within_route_log_probs).sum(dim=-1).mean()
        entropy = route_entropy + 0.5 * within_route_entropy
        ce_loss = F.cross_entropy(masked_logits, y_action)
        path_loss = logits.sum() * 0.0
        if bool(path_label_mask_local.any().item()) and ("path_logits" in outputs):
            path_loss = _masked_path_cross_entropy(
                outputs["path_logits"][path_label_mask_local],
                y_path_local[path_label_mask_local],
                path_valid_mask_local[path_label_mask_local],
            )
        return (
            policy_loss
            + q_loss
            + float(value_weight) * value_loss
            + float(policy_ce_weight) * (route_loss + ce_loss)
            + float(path_ce_weight) * path_loss
            - float(entropy_weight) * entropy
        )

    for _epoch in range(max(1, int(epochs))):
        model.train()
        outputs = model(x_train_n, path_x_train_n, path_valid_mask_train)
        loss = _actor_critic_loss(
            outputs,
            y_route_train,
            y_macro_train,
            y_reward_train_n,
            y_path_train,
            path_label_mask_train,
            path_valid_mask_train,
        )
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        opt.step()

        model.eval()
        with torch.no_grad():
            outputs_val = model(x_val_n, path_x_val_n, path_valid_mask_val)
            val_loss = _actor_critic_loss(
                outputs_val,
                y_route_val,
                y_macro_val,
                y_reward_val_n,
                y_path_val,
                path_label_mask_val,
                path_valid_mask_val,
            )
            val_loss_f = float(val_loss.item())
        if val_loss_f + 1.0e-8 < best_val:
            best_val = val_loss_f
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        train_out = model(x_train_n, path_x_train_n, path_valid_mask_train)
        val_out = model(x_val_n, path_x_val_n, path_valid_mask_val)
        train_q = train_out["q_values"] * reward_std + reward_mean
        val_q = val_out["q_values"] * reward_std + reward_mean
        train_value = train_out["value_pred"] * reward_std + reward_mean
        val_value = val_out["value_pred"] * reward_std + reward_mean
        train_metrics = _macro_metrics_from_logits(
            train_out["macro_logits"],
            y_macro_train,
            action_names=macro_action_names,
        )
        train_metrics.update(_hierarchical_macro_metrics(
            train_out["route_logits"],
            train_out["macro_logits"],
            y_route_train,
            y_macro_train,
            route_action_mask,
        ))
        train_metrics.update(_flat_classification_metrics(
            train_out["route_logits"],
            y_route_train,
            prefix="route",
        ))
        if bool(path_label_mask_train.any().item()):
            train_metrics.update(_path_metrics_from_logits(
                train_out["path_logits"][path_label_mask_train],
                y_path_train[path_label_mask_train],
                path_valid_mask_train[path_label_mask_train],
            ))
            train_metrics.update(_flat_classification_metrics(
                _gather_path_head(train_out["path_route_logits"][tuple_route_mask_train], y_path_train[tuple_route_mask_train]),
                y_route_train[tuple_route_mask_train],
                prefix="tuple_route",
            ))
            train_metrics.update(_hierarchical_macro_metrics(
                _gather_path_head(train_out["path_route_logits"][tuple_route_mask_train], y_path_train[tuple_route_mask_train]),
                _gather_path_head(train_out["path_macro_logits"][tuple_route_mask_train], y_path_train[tuple_route_mask_train]),
                y_route_train[tuple_route_mask_train],
                y_macro_train[tuple_route_mask_train],
                route_action_mask,
            ))
        train_q_chosen = train_q.gather(1, y_macro_train.reshape(-1, 1)).reshape(-1)
        train_metrics.update(_regression_metrics(train_q_chosen, y_reward_train, prefix="q_value"))
        if bool(tuple_route_mask_train.any().item()):
            train_q_tuple = _gather_path_head(train_out["path_q_values"][tuple_route_mask_train], y_path_train[tuple_route_mask_train]) * reward_std + reward_mean
            train_q_tuple_chosen = train_q_tuple.gather(1, y_macro_train[tuple_route_mask_train].reshape(-1, 1)).reshape(-1)
            train_metrics.update(_regression_metrics(train_q_tuple_chosen, y_reward_train[tuple_route_mask_train], prefix="tuple_q_value"))
        train_metrics.update(_regression_metrics(train_value, y_reward_train, prefix="value"))
        train_metrics["reward_mean"] = float(torch.mean(y_reward_train).item())
        train_metrics["reward_std"] = float(torch.std(y_reward_train, unbiased=False).item()) if y_reward_train.numel() > 0 else 0.0
        val_metrics = _macro_metrics_from_logits(
            val_out["macro_logits"],
            y_macro_val,
            action_names=macro_action_names,
        )
        val_metrics.update(_hierarchical_macro_metrics(
            val_out["route_logits"],
            val_out["macro_logits"],
            y_route_val,
            y_macro_val,
            route_action_mask,
        ))
        val_metrics.update(_flat_classification_metrics(
            val_out["route_logits"],
            y_route_val,
            prefix="route",
        ))
        if bool(path_label_mask_val.any().item()):
            val_metrics.update(_path_metrics_from_logits(
                val_out["path_logits"][path_label_mask_val],
                y_path_val[path_label_mask_val],
                path_valid_mask_val[path_label_mask_val],
            ))
            val_metrics.update(_flat_classification_metrics(
                _gather_path_head(val_out["path_route_logits"][tuple_route_mask_val], y_path_val[tuple_route_mask_val]),
                y_route_val[tuple_route_mask_val],
                prefix="tuple_route",
            ))
            val_metrics.update(_hierarchical_macro_metrics(
                _gather_path_head(val_out["path_route_logits"][tuple_route_mask_val], y_path_val[tuple_route_mask_val]),
                _gather_path_head(val_out["path_macro_logits"][tuple_route_mask_val], y_path_val[tuple_route_mask_val]),
                y_route_val[tuple_route_mask_val],
                y_macro_val[tuple_route_mask_val],
                route_action_mask,
            ))
        val_q_chosen = val_q.gather(1, y_macro_val.reshape(-1, 1)).reshape(-1)
        val_metrics.update(_regression_metrics(val_q_chosen, y_reward_val, prefix="q_value"))
        if bool(tuple_route_mask_val.any().item()):
            val_q_tuple = _gather_path_head(val_out["path_q_values"][tuple_route_mask_val], y_path_val[tuple_route_mask_val]) * reward_std + reward_mean
            val_q_tuple_chosen = val_q_tuple.gather(1, y_macro_val[tuple_route_mask_val].reshape(-1, 1)).reshape(-1)
            val_metrics.update(_regression_metrics(val_q_tuple_chosen, y_reward_val[tuple_route_mask_val], prefix="tuple_q_value"))
        val_metrics.update(_regression_metrics(val_value, y_reward_val, prefix="value"))
        val_metrics["reward_mean"] = float(torch.mean(y_reward_val).item())
        val_metrics["reward_std"] = float(torch.std(y_reward_val, unbiased=False).item()) if y_reward_val.numel() > 0 else 0.0

    utility_weights = dict(REPAIR_CRITIC_DEFAULT_UTILITY_WEIGHTS)
    if isinstance(init_bundle, dict) and isinstance(init_bundle.get("utility_weights"), dict):
        utility_weights.update(init_bundle["utility_weights"])
    return {
        "model_kind": REPAIR_CRITIC_SHARED_MODEL_KIND,
        "feature_names": feature_names,
        "path_feature_names": path_feature_names,
        "head_names": head_names,
        "macro_action_names": macro_action_names,
        "route_names": route_names,
        "mode_names": mode_names,
        "path_relation_names": relation_names,
        "hidden_dim": int(hidden_dim),
        "reward_per_s_scale": float(init_bundle.get("reward_per_s_scale", 1.0)) if isinstance(init_bundle, dict) else 1.0,
        "actor_critic_reward_target": str(reward_target or "descendant_preferred"),
        "actor_critic_reward_mean": float(reward_mean),
        "actor_critic_reward_std": float(reward_std),
        "feature_mean": mean.detach().cpu(),
        "feature_std": std.detach().cpu(),
        "path_feature_mean": torch.as_tensor(
            path_mean.detach().cpu(),
            dtype=torch.float32,
        ),
        "path_feature_std": torch.as_tensor(
            path_std.detach().cpu(),
            dtype=torch.float32,
        ),
        "utility_weights": utility_weights,
        "macro_head_trained": True,
        "route_head_trained": True,
        "q_head_trained": True,
        "value_head_trained": True,
        "path_head_trained": bool(path_label_mask.any().item()) or (bool(init_bundle.get("path_head_trained", False)) if isinstance(init_bundle, dict) else False),
        "path_action_head_trained": bool(path_label_mask.any().item()) or (bool(init_bundle.get("path_action_head_trained", False)) if isinstance(init_bundle, dict) else False),
        "path_relation_head_trained": bool(init_bundle.get("path_relation_head_trained", False)) if isinstance(init_bundle, dict) else False,
        "path_mode_head_trained": bool(init_bundle.get("path_mode_head_trained", False)) if isinstance(init_bundle, dict) else False,
        "path_improve_head_trained": bool(init_bundle.get("path_improve_head_trained", False)) if isinstance(init_bundle, dict) else False,
        "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "metrics": {
            "n_examples": int(len(training_rows)),
            "n_macro_examples": int(len(training_rows)),
            "n_route_examples": int(len(training_rows)),
            "n_path_context_examples": int(path_valid_mask.any(dim=1).sum().item()),
            "n_path_examples": int(path_label_mask.sum().item()),
            "n_train": int(x_train.shape[0]),
            "n_val": int(x_val.shape[0]),
            "best_val_loss": float(best_val),
            "reward_target": str(reward_target or "descendant_preferred"),
            "reward_source_counts": dict(reward_source_counts),
            "train": train_metrics,
            "val": val_metrics,
        },
    }


def train_repair_controller_slate_ranker(
    rows: Sequence[Any],
    *,
    hidden_dim: int = 32,
    epochs: int = 120,
    lr: float = 5.0e-3,
    weight_decay: float = 1.0e-4,
    val_fraction: float = 0.2,
    seed: int = 0,
    listwise_beta: float = 2.0,
    score_gap_floor: float = 1.0e-3,
    pairwise_gap_scale: float = 0.1,
    path_ce_weight: float = 0.20,
    route_aux_weight: float = 0.10,
    value_weight: float = 0.10,
    reward_target: str = "descendant_preferred",
    repair_action_names: Sequence[str] = ("inv_steer", "repair_option"),
    init_bundle: dict[str, Any] | None = None,
) -> dict[str, Any]:
    torch.manual_seed(int(seed))
    training_rows, reward_source_counts = _build_repair_slate_rows(
        rows,
        reward_target=reward_target,
        repair_action_names=repair_action_names,
        score_gap_floor=float(score_gap_floor),
    )
    if len(training_rows) < 8:
        raise ValueError("Need at least 8 repair-slate rows to train a slate ranker.")

    feature_names = list(REPAIR_CRITIC_FEATURE_NAMES)
    path_feature_names = list(REPAIR_CRITIC_PATH_FEATURE_NAMES)
    head_names = list(REPAIR_CRITIC_HEAD_NAMES)
    macro_action_names = list(REPAIR_CRITIC_MACRO_ACTION_NAMES)
    route_names = list(REPAIR_CRITIC_ROUTE_NAMES)
    mode_names = list(REPAIR_CRITIC_MODE_NAMES)
    relation_names = list(REPAIR_CRITIC_PATH_RELATION_NAMES)
    route_index = {name: idx for idx, name in enumerate(route_names)}
    repair_route_idx = int(route_index.get("repair", 0))
    configured_actions = tuple(
        name
        for name in macro_action_names
        if _normalize_action_name(name) in {_normalize_action_name(v) for v in repair_action_names}
    )

    n_rows = len(training_rows)
    x = torch.tensor(
        [[float(row["features"].get(name, 0.0)) for name in feature_names] for row in training_rows],
        dtype=torch.float32,
    )
    max_paths = max((len(tuple(row.get("path_rows", ()))) for row in training_rows), default=0)
    max_paths = max(1, int(max_paths))
    path_x = torch.zeros((n_rows, max_paths, len(path_feature_names)), dtype=torch.float32)
    path_valid_mask = torch.zeros((n_rows, max_paths), dtype=torch.bool)
    y_path = torch.full((n_rows,), -100, dtype=torch.long)
    path_label_mask = torch.zeros((n_rows,), dtype=torch.bool)
    y_path_utility = torch.zeros((n_rows, max_paths), dtype=torch.float32)
    path_utility_mask = torch.zeros((n_rows, max_paths), dtype=torch.bool)
    path_action_indices = torch.zeros((n_rows, max_paths), dtype=torch.long)
    full_slate_mask = torch.zeros((n_rows,), dtype=torch.bool)
    y_route = torch.full((n_rows,), int(repair_route_idx), dtype=torch.long)
    aux_reward = torch.zeros((n_rows,), dtype=torch.float32)
    aux_reward_mask = torch.zeros((n_rows,), dtype=torch.bool)
    pairwise_pairs: list[list[tuple[int, int, float]]] = []
    observed_action_names: set[str] = set()

    inv_steer_idx = int(macro_action_names.index("inv_steer")) if "inv_steer" in macro_action_names else 0
    for i, row in enumerate(training_rows):
        path_rows = tuple(row.get("path_rows", ()) or ())
        for j, path_row in enumerate(path_rows[:max_paths]):
            path_valid_mask[i, j] = True
            path_x[i, j] = torch.tensor(
                repair_path_feature_vector(path_row, feature_names=path_feature_names),
                dtype=torch.float32,
            )
        path_target_index = row.get("path_target_index", None)
        if isinstance(path_target_index, int) and 0 <= path_target_index < min(len(path_rows), max_paths):
            y_path[i] = int(path_target_index)
            path_label_mask[i] = True
        utility_targets = list(row.get("path_utility_targets", []) or [])
        utility_mask = list(row.get("path_utility_mask", []) or [])
        action_targets = list(row.get("path_action_indices", []) or [])
        for j in range(min(max_paths, len(utility_targets), len(utility_mask))):
            if bool(utility_mask[j]):
                path_utility_mask[i, j] = True
                y_path_utility[i, j] = float(utility_targets[j])
                action_idx = int(action_targets[j]) if j < len(action_targets) else int(inv_steer_idx)
                action_idx = max(0, min(len(macro_action_names) - 1, action_idx))
                path_action_indices[i, j] = int(action_idx)
                observed_action_names.add(str(macro_action_names[action_idx]))
            else:
                path_action_indices[i, j] = int(inv_steer_idx)
        full_slate_mask[i] = bool(row.get("full_slate", False))
        aux_reward_value = row.get("aux_reward", None)
        if aux_reward_value is not None and math.isfinite(float(aux_reward_value)):
            aux_reward[i] = float(aux_reward_value)
            aux_reward_mask[i] = True
        pairs_local: list[tuple[int, int, float]] = []
        for better_idx, worse_idx, gap in list(row.get("pairwise_pairs", []) or []):
            try:
                better = int(better_idx)
                worse = int(worse_idx)
                gap_f = float(gap)
            except Exception:
                continue
            if better < 0 or worse < 0 or better >= max_paths or worse >= max_paths:
                continue
            pairs_local.append((better, worse, gap_f))
        pairwise_pairs.append(pairs_local)

    train_idx, val_idx = _split_indices(n_rows, float(val_fraction), int(seed))
    x_train = x[train_idx]
    x_val = x[val_idx] if len(val_idx) > 0 else x_train
    path_x_train = path_x[train_idx]
    path_x_val = path_x[val_idx] if len(val_idx) > 0 else path_x_train
    path_valid_mask_train = path_valid_mask[train_idx]
    path_valid_mask_val = path_valid_mask[val_idx] if len(val_idx) > 0 else path_valid_mask_train
    y_path_train = y_path[train_idx]
    y_path_val = y_path[val_idx] if len(val_idx) > 0 else y_path_train
    path_label_mask_train = path_label_mask[train_idx]
    path_label_mask_val = path_label_mask[val_idx] if len(val_idx) > 0 else path_label_mask_train
    y_path_utility_train = y_path_utility[train_idx]
    y_path_utility_val = y_path_utility[val_idx] if len(val_idx) > 0 else y_path_utility_train
    path_utility_mask_train = path_utility_mask[train_idx]
    path_utility_mask_val = path_utility_mask[val_idx] if len(val_idx) > 0 else path_utility_mask_train
    path_action_indices_train = path_action_indices[train_idx]
    path_action_indices_val = path_action_indices[val_idx] if len(val_idx) > 0 else path_action_indices_train
    full_slate_mask_train = full_slate_mask[train_idx]
    full_slate_mask_val = full_slate_mask[val_idx] if len(val_idx) > 0 else full_slate_mask_train
    y_route_train = y_route[train_idx]
    y_route_val = y_route[val_idx] if len(val_idx) > 0 else y_route_train
    aux_reward_train = aux_reward[train_idx]
    aux_reward_val = aux_reward[val_idx] if len(val_idx) > 0 else aux_reward_train
    aux_reward_mask_train = aux_reward_mask[train_idx]
    aux_reward_mask_val = aux_reward_mask[val_idx] if len(val_idx) > 0 else aux_reward_mask_train
    pairwise_pairs_train = [pairwise_pairs[int(i)] for i in train_idx.tolist()]
    pairwise_pairs_val = [pairwise_pairs[int(i)] for i in val_idx.tolist()] if len(val_idx) > 0 else pairwise_pairs_train

    mean, std = _compute_feature_stats(x_train)
    x_train_n = _normalize_inputs(x_train, mean, std)
    x_val_n = _normalize_inputs(x_val, mean, std)
    if bool(path_valid_mask_train.any().item()):
        path_train_flat = path_x_train[path_valid_mask_train]
        path_mean, path_std = _compute_feature_stats(path_train_flat)
    elif isinstance(init_bundle, dict):
        path_mean = torch.as_tensor(
            init_bundle.get("path_feature_mean", torch.zeros(len(path_feature_names))),
            dtype=torch.float32,
        )
        path_std = torch.as_tensor(
            init_bundle.get("path_feature_std", torch.ones(len(path_feature_names))),
            dtype=torch.float32,
        )
    else:
        path_mean = torch.zeros((len(path_feature_names),), dtype=torch.float32)
        path_std = torch.ones((len(path_feature_names),), dtype=torch.float32)
    path_x_train_n = _normalize_inputs(path_x_train, path_mean.reshape(1, 1, -1), path_std.reshape(1, 1, -1))
    path_x_val_n = _normalize_inputs(path_x_val, path_mean.reshape(1, 1, -1), path_std.reshape(1, 1, -1))

    aux_reward_mean = float(aux_reward_train[aux_reward_mask_train].mean().item()) if bool(aux_reward_mask_train.any().item()) else 0.0
    aux_reward_std = (
        float(aux_reward_train[aux_reward_mask_train].std(unbiased=False).item())
        if bool(aux_reward_mask_train.any().item())
        else 1.0
    )
    aux_reward_std = max(1.0e-6, aux_reward_std)
    aux_reward_train_n = (aux_reward_train - aux_reward_mean) / aux_reward_std
    aux_reward_val_n = (aux_reward_val - aux_reward_mean) / aux_reward_std

    model = _RepairControllerSharedNet(
        int(x.shape[1]),
        int(hidden_dim),
        n_macro_actions=len(macro_action_names),
        n_routes=len(route_names),
        path_input_dim=len(path_feature_names),
    ).to(dtype=torch.float32)
    _maybe_init_model_from_bundle(model, init_bundle)
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))

    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_val = float("inf")
    patience = max(20, int(0.20 * max(1, int(epochs))))
    bad_epochs = 0

    def _slate_loss(
        outputs: dict[str, torch.Tensor],
        target_util: torch.Tensor,
        target_util_mask: torch.Tensor,
        target_action_idx: torch.Tensor,
        target_path: torch.Tensor,
        target_path_mask: torch.Tensor,
        target_full_mask: torch.Tensor,
        target_pairs: Sequence[Sequence[tuple[int, int, float]]],
        target_route: torch.Tensor,
        target_aux_reward: torch.Tensor,
        target_aux_reward_mask: torch.Tensor,
        target_path_valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        path_scores = _gather_path_action_scores(outputs["path_q_values"], target_action_idx)
        loss = outputs["path_q_values"].sum() * 0.0
        listwise_rows = target_full_mask & (target_util_mask.sum(dim=-1) >= 2)
        if bool(listwise_rows.any().item()):
            loss = loss + _listwise_slate_loss(
                path_scores[listwise_rows],
                target_util[listwise_rows],
                target_util_mask[listwise_rows],
                beta=float(listwise_beta),
                min_gap=float(score_gap_floor),
            )
        partial_rows = (~target_full_mask).to(torch.bool)
        partial_scores = path_scores[partial_rows]
        partial_pairs = [target_pairs[idx] for idx, keep in enumerate(partial_rows.tolist()) if bool(keep)]
        if partial_scores.shape[0] > 0:
            loss = loss + _pairwise_rank_loss(
                partial_scores,
                partial_pairs,
                gap_scale=float(pairwise_gap_scale),
            )
        if bool(target_path_mask.any().item()):
            loss = loss + float(path_ce_weight) * _masked_path_cross_entropy(
                outputs["path_logits"][target_path_mask],
                target_path[target_path_mask],
                target_path_valid_mask[target_path_mask],
            )
            tuple_route_logits = _gather_path_head(outputs["path_route_logits"][target_path_mask], target_path[target_path_mask])
            loss = loss + float(route_aux_weight) * F.cross_entropy(
                tuple_route_logits,
                target_route[target_path_mask],
            )
        loss = loss + float(route_aux_weight) * F.cross_entropy(outputs["route_logits"], target_route)
        if bool(target_aux_reward_mask.any().item()):
            loss = loss + float(value_weight) * F.smooth_l1_loss(
                outputs["value_pred"][target_aux_reward_mask],
                target_aux_reward[target_aux_reward_mask],
            )
        return loss

    for _epoch in range(max(1, int(epochs))):
        model.train()
        outputs = model(x_train_n, path_x_train_n, path_valid_mask_train)
        loss = _slate_loss(
            outputs,
            y_path_utility_train,
            path_utility_mask_train,
            path_action_indices_train,
            y_path_train,
            path_label_mask_train,
            full_slate_mask_train,
            pairwise_pairs_train,
            y_route_train,
            aux_reward_train_n,
            aux_reward_mask_train,
            path_valid_mask_train,
        )
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        opt.step()

        model.eval()
        with torch.no_grad():
            outputs_val = model(x_val_n, path_x_val_n, path_valid_mask_val)
            val_loss = _slate_loss(
                outputs_val,
                y_path_utility_val,
                path_utility_mask_val,
                path_action_indices_val,
                y_path_val,
                path_label_mask_val,
                full_slate_mask_val,
                pairwise_pairs_val,
                y_route_val,
                aux_reward_val_n,
                aux_reward_mask_val,
                path_valid_mask_val,
            )
            val_loss_f = float(val_loss.item())
        if val_loss_f + 1.0e-8 < best_val:
            best_val = val_loss_f
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        train_out = model(x_train_n, path_x_train_n, path_valid_mask_train)
        val_out = model(x_val_n, path_x_val_n, path_valid_mask_val)
        train_scores = _gather_path_action_scores(train_out["path_q_values"], path_action_indices_train)
        val_scores = _gather_path_action_scores(val_out["path_q_values"], path_action_indices_val)
        train_metrics = _slate_rank_metrics(
            train_scores,
            y_path_utility_train,
            path_utility_mask_train,
            pairwise_pairs_train,
            prefix="slate",
        )
        val_metrics = _slate_rank_metrics(
            val_scores,
            y_path_utility_val,
            path_utility_mask_val,
            pairwise_pairs_val,
            prefix="slate",
        )
        if bool(path_label_mask_train.any().item()):
            train_metrics.update(_path_metrics_from_logits(
                train_out["path_logits"][path_label_mask_train],
                y_path_train[path_label_mask_train],
                path_valid_mask_train[path_label_mask_train],
            ))
            train_metrics.update(_flat_classification_metrics(
                _gather_path_head(train_out["path_route_logits"][path_label_mask_train], y_path_train[path_label_mask_train]),
                y_route_train[path_label_mask_train],
                prefix="tuple_route",
            ))
        train_metrics.update(_flat_classification_metrics(
            train_out["route_logits"],
            y_route_train,
            prefix="route",
        ))
        if bool(aux_reward_mask_train.any().item()):
            train_value = train_out["value_pred"] * aux_reward_std + aux_reward_mean
            train_metrics.update(_regression_metrics(
                train_value[aux_reward_mask_train],
                aux_reward_train[aux_reward_mask_train],
                prefix="value",
            ))
        if bool(path_label_mask_val.any().item()):
            val_metrics.update(_path_metrics_from_logits(
                val_out["path_logits"][path_label_mask_val],
                y_path_val[path_label_mask_val],
                path_valid_mask_val[path_label_mask_val],
            ))
            val_metrics.update(_flat_classification_metrics(
                _gather_path_head(val_out["path_route_logits"][path_label_mask_val], y_path_val[path_label_mask_val]),
                y_route_val[path_label_mask_val],
                prefix="tuple_route",
            ))
        val_metrics.update(_flat_classification_metrics(
            val_out["route_logits"],
            y_route_val,
            prefix="route",
        ))
        if bool(aux_reward_mask_val.any().item()):
            val_value = val_out["value_pred"] * aux_reward_std + aux_reward_mean
            val_metrics.update(_regression_metrics(
                val_value[aux_reward_mask_val],
                aux_reward_val[aux_reward_mask_val],
                prefix="value",
            ))

    utility_weights = dict(REPAIR_CRITIC_DEFAULT_UTILITY_WEIGHTS)
    if isinstance(init_bundle, dict) and isinstance(init_bundle.get("utility_weights"), dict):
        utility_weights.update(init_bundle["utility_weights"])
    allowed_actions = tuple(
        name
        for name in configured_actions
        if name in observed_action_names
    ) or tuple(
        name
        for name in configured_actions
        if name == "inv_steer"
    ) or configured_actions
    return {
        "model_kind": REPAIR_CRITIC_SHARED_MODEL_KIND,
        "feature_names": feature_names,
        "path_feature_names": path_feature_names,
        "head_names": head_names,
        "macro_action_names": macro_action_names,
        "route_names": route_names,
        "mode_names": mode_names,
        "path_relation_names": relation_names,
        "hidden_dim": int(hidden_dim),
        "reward_per_s_scale": float(init_bundle.get("reward_per_s_scale", 1.0)) if isinstance(init_bundle, dict) else 1.0,
        "actor_critic_reward_target": str(reward_target or "descendant_preferred"),
        "actor_critic_reward_mean": float(aux_reward_mean),
        "actor_critic_reward_std": float(aux_reward_std),
        "feature_mean": mean.detach().cpu(),
        "feature_std": std.detach().cpu(),
        "path_feature_mean": torch.as_tensor(path_mean.detach().cpu(), dtype=torch.float32),
        "path_feature_std": torch.as_tensor(path_std.detach().cpu(), dtype=torch.float32),
        "utility_weights": utility_weights,
        "repair_slate_ranker_trained": True,
        "repair_slate_ranker_target": "same_state_exact_child_log_gain",
        "repair_slate_action_names": list(allowed_actions),
        "repair_slate_score_gap_floor": float(score_gap_floor),
        "repair_slate_listwise_beta": float(listwise_beta),
        "macro_head_trained": bool(init_bundle.get("macro_head_trained", False)) if isinstance(init_bundle, dict) else False,
        "route_head_trained": True,
        "q_head_trained": bool(init_bundle.get("q_head_trained", False)) if isinstance(init_bundle, dict) else False,
        "value_head_trained": bool(aux_reward_mask.any().item()),
        "path_head_trained": True,
        "path_action_head_trained": True,
        "path_relation_head_trained": bool(init_bundle.get("path_relation_head_trained", False)) if isinstance(init_bundle, dict) else False,
        "path_mode_head_trained": bool(init_bundle.get("path_mode_head_trained", False)) if isinstance(init_bundle, dict) else False,
        "path_improve_head_trained": bool(init_bundle.get("path_improve_head_trained", False)) if isinstance(init_bundle, dict) else False,
        "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "metrics": {
            "n_examples": int(len(training_rows)),
            "n_full_slate_examples": int(full_slate_mask.sum().item()),
            "n_pairwise_examples": int((~full_slate_mask).sum().item()),
            "n_pairwise_pairs": int(sum(len(pairs) for pairs in pairwise_pairs)),
            "n_path_examples": int(path_label_mask.sum().item()),
            "n_value_examples": int(aux_reward_mask.sum().item()),
            "n_train": int(x_train.shape[0]),
            "n_val": int(x_val.shape[0]),
            "best_val_loss": float(best_val),
            "reward_target": str(reward_target or "descendant_preferred"),
            "reward_source_counts": dict(reward_source_counts),
            "train": train_metrics,
            "val": val_metrics,
        },
    }


def train_repair_controller_tuple_ranker(
    rows: Sequence[Any],
    *,
    hidden_dim: int = 32,
    epochs: int = 120,
    lr: float = 5.0e-3,
    weight_decay: float = 1.0e-4,
    val_fraction: float = 0.2,
    seed: int = 0,
    listwise_beta: float = 2.0,
    score_gap_floor: float = 1.0e-3,
    pairwise_gap_scale: float = 0.1,
    path_ce_weight: float = 0.10,
    route_aux_weight: float = 0.05,
    preview_value_weight: float = 0.10,
    preview_regret_weight: float = 0.10,
    state_value_weight: float = 0.10,
    child_value_lambda: float = 0.25,
    reward_target: str = "descendant_preferred",
    repair_action_names: Sequence[str] = ("inv_steer", "repair_option"),
    init_bundle: dict[str, Any] | None = None,
) -> dict[str, Any]:
    torch.manual_seed(int(seed))
    training_rows, reward_source_counts = _build_repair_tuple_rows(
        rows,
        reward_target=reward_target,
        repair_action_names=repair_action_names,
        score_gap_floor=float(score_gap_floor),
    )
    if len(training_rows) < 8:
        raise ValueError("Need at least 8 repair-tuple rows to train a tuple critic.")

    feature_names = list(REPAIR_CRITIC_FEATURE_NAMES)
    path_feature_names = list(REPAIR_CRITIC_PATH_FEATURE_NAMES)
    preview_feature_names = list(REPAIR_CRITIC_PREVIEW_FEATURE_NAMES)
    provenance_feature_names = list(preview_feature_names)
    head_names = list(REPAIR_CRITIC_HEAD_NAMES)
    macro_action_names = list(REPAIR_CRITIC_MACRO_ACTION_NAMES)
    route_names = list(REPAIR_CRITIC_ROUTE_NAMES)
    mode_names = list(REPAIR_CRITIC_MODE_NAMES)
    relation_names = list(REPAIR_CRITIC_PATH_RELATION_NAMES)
    route_index = {name: idx for idx, name in enumerate(route_names)}
    repair_route_idx = int(route_index.get("repair", 0))
    configured_actions = tuple(
        name
        for name in macro_action_names
        if _normalize_action_name(name) in {_normalize_action_name(v) for v in repair_action_names}
    )

    n_rows = len(training_rows)
    x = torch.tensor(
        [[float(row["features"].get(name, 0.0)) for name in feature_names] for row in training_rows],
        dtype=torch.float32,
    )
    max_paths = max((len(tuple(row.get("path_rows", ()))) for row in training_rows), default=0)
    max_paths = max(1, int(max_paths))
    max_preview = max((len(list(row.get("preview_rows", []) or [])) for row in training_rows), default=0)
    max_preview = max(1, int(max_preview))
    max_provenance = max(
        (
            len(list(preview_row.get("provenance_rows", []) or [preview_row]))
            for row in training_rows
            for preview_row in list(row.get("preview_rows", []) or [])
            if isinstance(preview_row, Mapping)
        ),
        default=0,
    )
    max_provenance = max(1, int(max_provenance))

    path_x = torch.zeros((n_rows, max_paths, len(path_feature_names)), dtype=torch.float32)
    path_valid_mask = torch.zeros((n_rows, max_paths), dtype=torch.bool)
    y_path = torch.full((n_rows,), -100, dtype=torch.long)
    path_label_mask = torch.zeros((n_rows,), dtype=torch.bool)
    preview_x = torch.zeros((n_rows, max_preview, len(preview_feature_names)), dtype=torch.float32)
    preview_mask = torch.zeros((n_rows, max_preview), dtype=torch.bool)
    preview_path_indices = torch.zeros((n_rows, max_preview), dtype=torch.long)
    provenance_x = torch.zeros((n_rows, max_preview, max_provenance, len(provenance_feature_names)), dtype=torch.float32)
    provenance_mask = torch.zeros((n_rows, max_preview, max_provenance), dtype=torch.bool)
    y_preview_utility = torch.zeros((n_rows, max_preview), dtype=torch.float32)
    y_preview_value = torch.zeros((n_rows, max_preview), dtype=torch.float32)
    y_preview_regret = torch.zeros((n_rows, max_preview), dtype=torch.float32)
    preview_value_mask = torch.zeros((n_rows, max_preview), dtype=torch.bool)
    full_slate_mask = torch.zeros((n_rows,), dtype=torch.bool)
    y_route = torch.full((n_rows,), int(repair_route_idx), dtype=torch.long)
    state_value_target = torch.zeros((n_rows,), dtype=torch.float32)
    pairwise_pairs: list[list[tuple[int, int, float]]] = []
    observed_action_names: set[str] = set()

    for i, row in enumerate(training_rows):
        path_rows = tuple(row.get("path_rows", ()) or ())
        for j, path_row in enumerate(path_rows[:max_paths]):
            path_valid_mask[i, j] = True
            path_x[i, j] = torch.tensor(
                repair_path_feature_vector(path_row, feature_names=path_feature_names),
                dtype=torch.float32,
            )
        path_target_index = row.get("path_target_index", None)
        if isinstance(path_target_index, int) and 0 <= path_target_index < min(len(path_rows), max_paths):
            y_path[i] = int(path_target_index)
            path_label_mask[i] = True
        preview_rows = list(row.get("preview_rows", []) or [])
        preview_path_idx_list = list(row.get("preview_path_indices", []) or [])
        preview_utilities = list(row.get("preview_utility_targets", []) or [])
        preview_values = list(row.get("preview_value_targets", []) or [])
        preview_regrets = list(row.get("preview_regret_targets", []) or [])
        preview_value_flags = list(row.get("preview_value_mask", []) or [])
        for j in range(min(max_preview, len(preview_rows), len(preview_path_idx_list), len(preview_utilities))):
            preview_mask[i, j] = True
            preview_x[i, j] = torch.tensor(
                repair_preview_feature_vector(preview_rows[j], feature_names=preview_feature_names),
                dtype=torch.float32,
            )
            preview_path_indices[i, j] = int(max(0, min(max_paths - 1, int(preview_path_idx_list[j]))))
            provenance_rows = list(preview_rows[j].get("provenance_rows", []) or [preview_rows[j]])
            for k in range(min(max_provenance, len(provenance_rows))):
                provenance_mask[i, j, k] = True
                provenance_x[i, j, k] = torch.tensor(
                    repair_preview_feature_vector(provenance_rows[k], feature_names=provenance_feature_names),
                    dtype=torch.float32,
                )
            y_preview_utility[i, j] = float(preview_utilities[j])
            if j < len(preview_values):
                y_preview_value[i, j] = float(preview_values[j])
            if j < len(preview_regrets):
                y_preview_regret[i, j] = max(0.0, float(preview_regrets[j]))
            if j < len(preview_value_flags):
                preview_value_mask[i, j] = bool(preview_value_flags[j])
            observed_action_names.add(_normalize_action_name(preview_rows[j].get("action", "inv_steer")))
        full_slate_mask[i] = bool(row.get("full_slate", False))
        state_value_target[i] = float(row.get("state_value_target", 0.0) or 0.0)
        pairs_local: list[tuple[int, int, float]] = []
        for better_idx, worse_idx, gap in list(row.get("pairwise_pairs", []) or []):
            try:
                better = int(better_idx)
                worse = int(worse_idx)
                gap_f = float(gap)
            except Exception:
                continue
            if better < 0 or worse < 0 or better >= max_preview or worse >= max_preview:
                continue
            pairs_local.append((better, worse, gap_f))
        pairwise_pairs.append(pairs_local)

    train_idx, val_idx = _split_indices(n_rows, float(val_fraction), int(seed))
    x_train = x[train_idx]
    x_val = x[val_idx] if len(val_idx) > 0 else x_train
    path_x_train = path_x[train_idx]
    path_x_val = path_x[val_idx] if len(val_idx) > 0 else path_x_train
    path_valid_mask_train = path_valid_mask[train_idx]
    path_valid_mask_val = path_valid_mask[val_idx] if len(val_idx) > 0 else path_valid_mask_train
    y_path_train = y_path[train_idx]
    y_path_val = y_path[val_idx] if len(val_idx) > 0 else y_path_train
    path_label_mask_train = path_label_mask[train_idx]
    path_label_mask_val = path_label_mask[val_idx] if len(val_idx) > 0 else path_label_mask_train
    preview_x_train = preview_x[train_idx]
    preview_x_val = preview_x[val_idx] if len(val_idx) > 0 else preview_x_train
    preview_mask_train = preview_mask[train_idx]
    preview_mask_val = preview_mask[val_idx] if len(val_idx) > 0 else preview_mask_train
    preview_path_indices_train = preview_path_indices[train_idx]
    preview_path_indices_val = preview_path_indices[val_idx] if len(val_idx) > 0 else preview_path_indices_train
    provenance_x_train = provenance_x[train_idx]
    provenance_x_val = provenance_x[val_idx] if len(val_idx) > 0 else provenance_x_train
    provenance_mask_train = provenance_mask[train_idx]
    provenance_mask_val = provenance_mask[val_idx] if len(val_idx) > 0 else provenance_mask_train
    y_preview_utility_train = y_preview_utility[train_idx]
    y_preview_utility_val = y_preview_utility[val_idx] if len(val_idx) > 0 else y_preview_utility_train
    y_preview_value_train = y_preview_value[train_idx]
    y_preview_value_val = y_preview_value[val_idx] if len(val_idx) > 0 else y_preview_value_train
    y_preview_regret_train = y_preview_regret[train_idx]
    y_preview_regret_val = y_preview_regret[val_idx] if len(val_idx) > 0 else y_preview_regret_train
    preview_value_mask_train = preview_value_mask[train_idx]
    preview_value_mask_val = preview_value_mask[val_idx] if len(val_idx) > 0 else preview_value_mask_train
    full_slate_mask_train = full_slate_mask[train_idx]
    full_slate_mask_val = full_slate_mask[val_idx] if len(val_idx) > 0 else full_slate_mask_train
    y_route_train = y_route[train_idx]
    y_route_val = y_route[val_idx] if len(val_idx) > 0 else y_route_train
    state_value_target_train = state_value_target[train_idx]
    state_value_target_val = state_value_target[val_idx] if len(val_idx) > 0 else state_value_target_train
    pairwise_pairs_train = [pairwise_pairs[int(i)] for i in train_idx.tolist()]
    pairwise_pairs_val = [pairwise_pairs[int(i)] for i in val_idx.tolist()] if len(val_idx) > 0 else pairwise_pairs_train

    mean, std = _compute_feature_stats(x_train)
    x_train_n = _normalize_inputs(x_train, mean, std)
    x_val_n = _normalize_inputs(x_val, mean, std)
    if bool(path_valid_mask_train.any().item()):
        path_train_flat = path_x_train[path_valid_mask_train]
        path_mean, path_std = _compute_feature_stats(path_train_flat)
    elif isinstance(init_bundle, dict):
        path_mean = torch.as_tensor(
            init_bundle.get("path_feature_mean", torch.zeros(len(path_feature_names))),
            dtype=torch.float32,
        )
        path_std = torch.as_tensor(
            init_bundle.get("path_feature_std", torch.ones(len(path_feature_names))),
            dtype=torch.float32,
        )
    else:
        path_mean = torch.zeros((len(path_feature_names),), dtype=torch.float32)
        path_std = torch.ones((len(path_feature_names),), dtype=torch.float32)
    path_x_train_n = _normalize_inputs(path_x_train, path_mean.reshape(1, 1, -1), path_std.reshape(1, 1, -1))
    path_x_val_n = _normalize_inputs(path_x_val, path_mean.reshape(1, 1, -1), path_std.reshape(1, 1, -1))

    if bool(preview_mask_train.any().item()):
        preview_train_flat = preview_x_train[preview_mask_train]
        preview_mean, preview_std = _compute_feature_stats(preview_train_flat)
    else:
        preview_mean = torch.zeros((len(preview_feature_names),), dtype=torch.float32)
        preview_std = torch.ones((len(preview_feature_names),), dtype=torch.float32)
    preview_x_train_n = _normalize_inputs(preview_x_train, preview_mean.reshape(1, 1, -1), preview_std.reshape(1, 1, -1))
    preview_x_val_n = _normalize_inputs(preview_x_val, preview_mean.reshape(1, 1, -1), preview_std.reshape(1, 1, -1))
    if bool(provenance_mask_train.any().item()):
        provenance_train_flat = provenance_x_train[provenance_mask_train]
        provenance_mean, provenance_std = _compute_feature_stats(provenance_train_flat)
    else:
        provenance_mean = torch.zeros((len(provenance_feature_names),), dtype=torch.float32)
        provenance_std = torch.ones((len(provenance_feature_names),), dtype=torch.float32)
    provenance_x_train_n = _normalize_inputs(
        provenance_x_train,
        provenance_mean.reshape(1, 1, 1, -1),
        provenance_std.reshape(1, 1, 1, -1),
    )
    provenance_x_val_n = _normalize_inputs(
        provenance_x_val,
        provenance_mean.reshape(1, 1, 1, -1),
        provenance_std.reshape(1, 1, 1, -1),
    )

    model = _RepairControllerSharedNet(
        int(x.shape[1]),
        int(hidden_dim),
        n_macro_actions=len(macro_action_names),
        n_routes=len(route_names),
        path_input_dim=len(path_feature_names),
        preview_input_dim=len(preview_feature_names),
        provenance_input_dim=len(provenance_feature_names),
    ).to(dtype=torch.float32)
    _maybe_init_model_from_bundle(model, init_bundle)
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))

    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_val = float("inf")
    patience = max(20, int(0.20 * max(1, int(epochs))))
    bad_epochs = 0

    def _tuple_loss(
        outputs: dict[str, torch.Tensor],
        target_preview_utility: torch.Tensor,
        target_preview_value: torch.Tensor,
        target_preview_regret: torch.Tensor,
        target_preview_mask: torch.Tensor,
        target_preview_value_mask: torch.Tensor,
        target_path: torch.Tensor,
        target_path_mask: torch.Tensor,
        target_full_mask: torch.Tensor,
        target_pairs: Sequence[Sequence[tuple[int, int, float]]],
        target_route: torch.Tensor,
        target_state_value: torch.Tensor,
        target_path_valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        preview_scores = outputs["preview_utility"]
        loss = preview_scores.sum() * 0.0
        listwise_rows = target_full_mask & (target_preview_mask.sum(dim=-1) >= 2)
        if bool(listwise_rows.any().item()):
            loss = loss + _listwise_slate_loss(
                preview_scores[listwise_rows],
                target_preview_utility[listwise_rows],
                target_preview_mask[listwise_rows],
                beta=float(listwise_beta),
                min_gap=float(score_gap_floor),
            )
        partial_rows = (~target_full_mask) & (target_preview_mask.sum(dim=-1) >= 2)
        partial_scores = preview_scores[partial_rows]
        partial_pairs = [target_pairs[idx] for idx, keep in enumerate(partial_rows.tolist()) if bool(keep)]
        if partial_scores.shape[0] > 0:
            loss = loss + _pairwise_rank_loss(
                partial_scores,
                partial_pairs,
                gap_scale=float(pairwise_gap_scale),
            )
        if bool(target_path_mask.any().item()):
            loss = loss + float(path_ce_weight) * _masked_path_cross_entropy(
                outputs["path_logits"][target_path_mask],
                target_path[target_path_mask],
                target_path_valid_mask[target_path_mask],
            )
        if float(route_aux_weight) > 0.0:
            loss = loss + float(route_aux_weight) * F.cross_entropy(outputs["route_logits"], target_route)
        if float(preview_value_weight) > 0.0 and bool(target_preview_value_mask.any().item()):
            loss = loss + float(preview_value_weight) * F.smooth_l1_loss(
                outputs["preview_value"][target_preview_value_mask],
                target_preview_value[target_preview_value_mask],
            )
        if float(preview_regret_weight) > 0.0 and bool(target_preview_mask.any().item()):
            loss = loss + float(preview_regret_weight) * F.smooth_l1_loss(
                outputs["preview_regret"][target_preview_mask],
                target_preview_regret[target_preview_mask],
            )
        if float(state_value_weight) > 0.0:
            state_mask = target_preview_mask.any(dim=-1)
            loss = loss + float(state_value_weight) * F.smooth_l1_loss(
                outputs["value_pred"][state_mask],
                target_state_value[state_mask],
            )
        return loss

    for _epoch in range(max(1, int(epochs))):
        model.train()
        outputs = model(
            x_train_n,
            path_x_train_n,
            path_valid_mask_train,
            preview_x_train_n,
            preview_path_indices_train,
            preview_mask_train,
            provenance_x_train_n,
            provenance_mask_train,
        )
        loss = _tuple_loss(
            outputs,
            y_preview_utility_train,
            y_preview_value_train,
            y_preview_regret_train,
            preview_mask_train,
            preview_value_mask_train,
            y_path_train,
            path_label_mask_train,
            full_slate_mask_train,
            pairwise_pairs_train,
            y_route_train,
            state_value_target_train,
            path_valid_mask_train,
        )
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        opt.step()

        model.eval()
        with torch.no_grad():
            outputs_val = model(
                x_val_n,
                path_x_val_n,
                path_valid_mask_val,
                preview_x_val_n,
                preview_path_indices_val,
                preview_mask_val,
                provenance_x_val_n,
                provenance_mask_val,
            )
            val_loss = _tuple_loss(
                outputs_val,
                y_preview_utility_val,
                y_preview_value_val,
                y_preview_regret_val,
                preview_mask_val,
                preview_value_mask_val,
                y_path_val,
                path_label_mask_val,
                full_slate_mask_val,
                pairwise_pairs_val,
                y_route_val,
                state_value_target_val,
                path_valid_mask_val,
            )
            val_loss_f = float(val_loss.item())
        if val_loss_f + 1.0e-8 < best_val:
            best_val = val_loss_f
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        train_out = model(
            x_train_n,
            path_x_train_n,
            path_valid_mask_train,
            preview_x_train_n,
            preview_path_indices_train,
            preview_mask_train,
            provenance_x_train_n,
            provenance_mask_train,
        )
        val_out = model(
            x_val_n,
            path_x_val_n,
            path_valid_mask_val,
            preview_x_val_n,
            preview_path_indices_val,
            preview_mask_val,
            provenance_x_val_n,
            provenance_mask_val,
        )
        train_metrics = _slate_rank_metrics(
            train_out["preview_utility"],
            y_preview_utility_train,
            preview_mask_train,
            pairwise_pairs_train,
            prefix="tuple",
        )
        val_metrics = _slate_rank_metrics(
            val_out["preview_utility"],
            y_preview_utility_val,
            preview_mask_val,
            pairwise_pairs_val,
            prefix="tuple",
        )
        if bool(path_label_mask_train.any().item()):
            train_metrics.update(_path_metrics_from_logits(
                train_out["path_logits"][path_label_mask_train],
                y_path_train[path_label_mask_train],
                path_valid_mask_train[path_label_mask_train],
            ))
        if bool(path_label_mask_val.any().item()):
            val_metrics.update(_path_metrics_from_logits(
                val_out["path_logits"][path_label_mask_val],
                y_path_val[path_label_mask_val],
                path_valid_mask_val[path_label_mask_val],
            ))
        if float(route_aux_weight) > 0.0:
            train_metrics.update(_flat_classification_metrics(
                train_out["route_logits"],
                y_route_train,
                prefix="route",
            ))
            val_metrics.update(_flat_classification_metrics(
                val_out["route_logits"],
                y_route_val,
                prefix="route",
            ))
        if bool(preview_value_mask_train.any().item()):
            train_metrics.update(_regression_metrics(
                train_out["preview_value"][preview_value_mask_train],
                y_preview_value_train[preview_value_mask_train],
                prefix="preview_value",
            ))
        if bool(preview_value_mask_val.any().item()):
            val_metrics.update(_regression_metrics(
                val_out["preview_value"][preview_value_mask_val],
                y_preview_value_val[preview_value_mask_val],
                prefix="preview_value",
            ))
        if bool(preview_mask_train.any().item()):
            train_metrics.update(_regression_metrics(
                train_out["preview_regret"][preview_mask_train],
                y_preview_regret_train[preview_mask_train],
                prefix="preview_regret",
            ))
        if bool(preview_mask_val.any().item()):
            val_metrics.update(_regression_metrics(
                val_out["preview_regret"][preview_mask_val],
                y_preview_regret_val[preview_mask_val],
                prefix="preview_regret",
            ))
        state_mask_train = preview_mask_train.any(dim=-1)
        state_mask_val = preview_mask_val.any(dim=-1)
        if bool(state_mask_train.any().item()):
            train_metrics.update(_regression_metrics(
                train_out["value_pred"][state_mask_train],
                state_value_target_train[state_mask_train],
                prefix="state_value",
            ))
        if bool(state_mask_val.any().item()):
            val_metrics.update(_regression_metrics(
                val_out["value_pred"][state_mask_val],
                state_value_target_val[state_mask_val],
                prefix="state_value",
            ))

    return {
        "model_kind": REPAIR_CRITIC_SHARED_MODEL_KIND,
        "feature_names": feature_names,
        "path_feature_names": path_feature_names,
        "preview_feature_names": preview_feature_names,
        "provenance_feature_names": provenance_feature_names,
        "head_names": head_names,
        "macro_action_names": macro_action_names,
        "route_names": route_names,
        "mode_names": mode_names,
        "path_relation_names": relation_names,
        "hidden_dim": int(hidden_dim),
        "reward_per_s_scale": float(init_bundle.get("reward_per_s_scale", 1.0)) if isinstance(init_bundle, dict) else 1.0,
        "actor_critic_reward_target": str(reward_target or "descendant_preferred"),
        "actor_critic_reward_mean": 0.0,
        "actor_critic_reward_std": 1.0,
        "feature_mean": mean.detach().cpu(),
        "feature_std": std.detach().cpu(),
        "path_feature_mean": torch.as_tensor(path_mean.detach().cpu(), dtype=torch.float32),
        "path_feature_std": torch.as_tensor(path_std.detach().cpu(), dtype=torch.float32),
        "preview_feature_mean": torch.as_tensor(preview_mean.detach().cpu(), dtype=torch.float32),
        "preview_feature_std": torch.as_tensor(preview_std.detach().cpu(), dtype=torch.float32),
        "provenance_feature_mean": torch.as_tensor(provenance_mean.detach().cpu(), dtype=torch.float32),
        "provenance_feature_std": torch.as_tensor(provenance_std.detach().cpu(), dtype=torch.float32),
        "utility_weights": dict(REPAIR_CRITIC_DEFAULT_UTILITY_WEIGHTS),
        "repair_tuple_ranker_trained": True,
        "repair_tuple_ranker_target": "same_state_exact_child_log_gain",
        "repair_tuple_preview_value_target": "child_descendant_log_gain",
        "repair_tuple_regret_target": "same_state_best_exact_regret",
        "repair_tuple_child_value_lambda": float(max(0.0, child_value_lambda)),
        "repair_tuple_regret_weight": float(max(0.0, preview_regret_weight)),
        "repair_tuple_action_names": list(
            name
            for name in configured_actions
            if _normalize_action_name(name) in observed_action_names
        ) or list(configured_actions),
        "repair_slate_ranker_trained": bool(init_bundle.get("repair_slate_ranker_trained", False)) if isinstance(init_bundle, dict) else False,
        "repair_slate_ranker_target": str(init_bundle.get("repair_slate_ranker_target", "")) if isinstance(init_bundle, dict) else "",
        "repair_slate_action_names": list(init_bundle.get("repair_slate_action_names", [])) if isinstance(init_bundle, dict) else [],
        "macro_head_trained": bool(init_bundle.get("macro_head_trained", False)) if isinstance(init_bundle, dict) else False,
        "route_head_trained": float(route_aux_weight) > 0.0,
        "q_head_trained": bool(init_bundle.get("q_head_trained", False)) if isinstance(init_bundle, dict) else False,
        "value_head_trained": True,
        "regret_head_trained": True,
        "path_head_trained": True,
        "path_action_head_trained": False,
        "path_relation_head_trained": bool(init_bundle.get("path_relation_head_trained", False)) if isinstance(init_bundle, dict) else False,
        "path_mode_head_trained": bool(init_bundle.get("path_mode_head_trained", False)) if isinstance(init_bundle, dict) else False,
        "path_improve_head_trained": bool(init_bundle.get("path_improve_head_trained", False)) if isinstance(init_bundle, dict) else False,
        "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "metrics": {
            "n_examples": int(len(training_rows)),
            "n_full_slate_examples": int(full_slate_mask.sum().item()),
            "n_pairwise_examples": int((~full_slate_mask).sum().item()),
            "n_pairwise_pairs": int(sum(len(pairs) for pairs in pairwise_pairs)),
            "n_path_examples": int(path_label_mask.sum().item()),
            "n_preview_examples": int(preview_mask.any(dim=-1).sum().item()),
            "n_tuple_examples": int(preview_mask.sum().item()),
            "n_provenance_tuples": int(provenance_mask.sum().item()),
            "n_preview_value_examples": int(preview_value_mask.any(dim=-1).sum().item()),
            "n_preview_value_tuples": int(preview_value_mask.sum().item()),
            "n_preview_regret_examples": int(preview_mask.any(dim=-1).sum().item()),
            "n_preview_regret_tuples": int(preview_mask.sum().item()),
            "n_train": int(x_train.shape[0]),
            "n_val": int(x_val.shape[0]),
            "best_val_loss": float(best_val),
            "reward_target": str(reward_target or "descendant_preferred"),
            "reward_source_counts": dict(reward_source_counts),
            "train": train_metrics,
            "val": val_metrics,
        },
    }


def train_unified_candidate_ranker(
    rows: Sequence[Any],
    *,
    hidden_dim: int = 32,
    epochs: int = 120,
    lr: float = 5.0e-3,
    weight_decay: float = 1.0e-4,
    val_fraction: float = 0.2,
    seed: int = 0,
    listwise_beta: float = 2.0,
    score_gap_floor: float = 1.0e-3,
    pairwise_gap_scale: float = 0.1,
    path_ce_weight: float = 0.05,
    route_aux_weight: float = 0.0,
    preview_regret_weight: float = 0.10,
    state_value_weight: float = 0.10,
    repair_action_names: Sequence[str] = ("inv_steer", "repair_option"),
    build_action_names: Sequence[str] = ("replace", "wrap_un", "residual"),
    init_bundle: dict[str, Any] | None = None,
) -> dict[str, Any]:
    torch.manual_seed(int(seed))
    training_rows, reward_source_counts = _build_unified_candidate_rows(
        rows,
        score_gap_floor=float(score_gap_floor),
        repair_action_names=repair_action_names,
        build_action_names=build_action_names,
    )
    if len(training_rows) < 8:
        raise ValueError("Need at least 8 unified candidate rows to train a shared continuation critic.")

    feature_names = list(REPAIR_CRITIC_FEATURE_NAMES)
    path_feature_names = list(REPAIR_CRITIC_PATH_FEATURE_NAMES)
    preview_feature_names = list(UNIFIED_CANDIDATE_PREVIEW_FEATURE_NAMES)
    provenance_feature_names = list(preview_feature_names)
    head_names = list(REPAIR_CRITIC_HEAD_NAMES)
    macro_action_names = list(REPAIR_CRITIC_MACRO_ACTION_NAMES)
    route_names = list(REPAIR_CRITIC_ROUTE_NAMES)
    mode_names = list(REPAIR_CRITIC_MODE_NAMES)
    relation_names = list(REPAIR_CRITIC_PATH_RELATION_NAMES)
    route_index = {name: idx for idx, name in enumerate(route_names)}

    n_rows = len(training_rows)
    x = torch.tensor(
        [[float(row["features"].get(name, 0.0)) for name in feature_names] for row in training_rows],
        dtype=torch.float32,
    )
    max_paths = max((len(tuple(row.get("path_rows", ()))) for row in training_rows), default=0)
    max_paths = max(1, int(max_paths))
    max_preview = max((len(list(row.get("preview_rows", []) or [])) for row in training_rows), default=0)
    max_preview = max(1, int(max_preview))
    max_provenance = max(
        (
            len(list(preview_row.get("provenance_rows", []) or [preview_row]))
            for row in training_rows
            for preview_row in list(row.get("preview_rows", []) or [])
            if isinstance(preview_row, Mapping)
        ),
        default=0,
    )
    max_provenance = max(1, int(max_provenance))

    path_x = torch.zeros((n_rows, max_paths, len(path_feature_names)), dtype=torch.float32)
    path_valid_mask = torch.zeros((n_rows, max_paths), dtype=torch.bool)
    y_path = torch.full((n_rows,), -100, dtype=torch.long)
    path_label_mask = torch.zeros((n_rows,), dtype=torch.bool)
    preview_x = torch.zeros((n_rows, max_preview, len(preview_feature_names)), dtype=torch.float32)
    preview_mask = torch.zeros((n_rows, max_preview), dtype=torch.bool)
    preview_path_indices = torch.zeros((n_rows, max_preview), dtype=torch.long)
    provenance_x = torch.zeros((n_rows, max_preview, max_provenance, len(provenance_feature_names)), dtype=torch.float32)
    provenance_mask = torch.zeros((n_rows, max_preview, max_provenance), dtype=torch.bool)
    y_preview_utility = torch.zeros((n_rows, max_preview), dtype=torch.float32)
    y_preview_regret = torch.zeros((n_rows, max_preview), dtype=torch.float32)
    full_slate_mask = torch.zeros((n_rows,), dtype=torch.bool)
    y_route = torch.zeros((n_rows,), dtype=torch.long)
    state_value_target = torch.zeros((n_rows,), dtype=torch.float32)
    pairwise_pairs: list[list[tuple[int, int, float]]] = []
    preview_route_targets: list[list[str]] = []
    observed_action_names: set[str] = set()

    for i, row in enumerate(training_rows):
        path_rows = tuple(row.get("path_rows", ()) or ())
        for j, path_row in enumerate(path_rows[:max_paths]):
            path_valid_mask[i, j] = True
            path_x[i, j] = torch.tensor(
                repair_path_feature_vector(path_row, feature_names=path_feature_names),
                dtype=torch.float32,
            )
        path_target_index = row.get("path_target_index", None)
        if isinstance(path_target_index, int) and 0 <= path_target_index < min(len(path_rows), max_paths):
            y_path[i] = int(path_target_index)
            path_label_mask[i] = True
        route_name = str(row.get("route_target", "build") or "build")
        y_route[i] = int(route_index.get(route_name, route_index.get("build", 0)))
        preview_rows = list(row.get("preview_rows", []) or [])
        preview_path_idx_list = list(row.get("preview_path_indices", []) or [])
        preview_utilities = list(row.get("preview_utility_targets", []) or [])
        preview_regrets = list(row.get("preview_regret_targets", []) or [])
        preview_routes = list(row.get("preview_route_targets", []) or [])
        route_rows_local: list[str] = []
        for j in range(min(max_preview, len(preview_rows), len(preview_path_idx_list), len(preview_utilities))):
            preview_mask[i, j] = True
            preview_x[i, j] = torch.tensor(
                unified_candidate_preview_feature_vector(preview_rows[j], feature_names=preview_feature_names),
                dtype=torch.float32,
            )
            preview_path_indices[i, j] = int(max(0, min(max_paths - 1, int(preview_path_idx_list[j]))))
            provenance_rows = list(preview_rows[j].get("provenance_rows", []) or [preview_rows[j]])
            for k in range(min(max_provenance, len(provenance_rows))):
                provenance_mask[i, j, k] = True
                provenance_x[i, j, k] = torch.tensor(
                    unified_candidate_preview_feature_vector(provenance_rows[k], feature_names=provenance_feature_names),
                    dtype=torch.float32,
                )
            y_preview_utility[i, j] = float(preview_utilities[j])
            if j < len(preview_regrets):
                y_preview_regret[i, j] = max(0.0, float(preview_regrets[j]))
            route_rows_local.append(str(preview_routes[j] if j < len(preview_routes) else preview_rows[j].get("route_source", "")))
            observed_action_names.add(_normalize_action_name(preview_rows[j].get("action", "")))
        preview_route_targets.append(route_rows_local)
        full_slate_mask[i] = bool(row.get("full_slate", False))
        state_value_target[i] = float(row.get("state_value_target", 0.0) or 0.0)
        pairs_local: list[tuple[int, int, float]] = []
        for better_idx, worse_idx, gap in list(row.get("pairwise_pairs", []) or []):
            try:
                better = int(better_idx)
                worse = int(worse_idx)
                gap_f = float(gap)
            except Exception:
                continue
            if better < 0 or worse < 0 or better >= max_preview or worse >= max_preview:
                continue
            pairs_local.append((better, worse, gap_f))
        pairwise_pairs.append(pairs_local)

    train_idx, val_idx = _split_indices(n_rows, float(val_fraction), int(seed))
    x_train = x[train_idx]
    x_val = x[val_idx] if len(val_idx) > 0 else x_train
    path_x_train = path_x[train_idx]
    path_x_val = path_x[val_idx] if len(val_idx) > 0 else path_x_train
    path_valid_mask_train = path_valid_mask[train_idx]
    path_valid_mask_val = path_valid_mask[val_idx] if len(val_idx) > 0 else path_valid_mask_train
    y_path_train = y_path[train_idx]
    y_path_val = y_path[val_idx] if len(val_idx) > 0 else y_path_train
    path_label_mask_train = path_label_mask[train_idx]
    path_label_mask_val = path_label_mask[val_idx] if len(val_idx) > 0 else path_label_mask_train
    preview_x_train = preview_x[train_idx]
    preview_x_val = preview_x[val_idx] if len(val_idx) > 0 else preview_x_train
    preview_mask_train = preview_mask[train_idx]
    preview_mask_val = preview_mask[val_idx] if len(val_idx) > 0 else preview_mask_train
    preview_path_indices_train = preview_path_indices[train_idx]
    preview_path_indices_val = preview_path_indices[val_idx] if len(val_idx) > 0 else preview_path_indices_train
    provenance_x_train = provenance_x[train_idx]
    provenance_x_val = provenance_x[val_idx] if len(val_idx) > 0 else provenance_x_train
    provenance_mask_train = provenance_mask[train_idx]
    provenance_mask_val = provenance_mask[val_idx] if len(val_idx) > 0 else provenance_mask_train
    y_preview_utility_train = y_preview_utility[train_idx]
    y_preview_utility_val = y_preview_utility[val_idx] if len(val_idx) > 0 else y_preview_utility_train
    y_preview_regret_train = y_preview_regret[train_idx]
    y_preview_regret_val = y_preview_regret[val_idx] if len(val_idx) > 0 else y_preview_regret_train
    full_slate_mask_train = full_slate_mask[train_idx]
    full_slate_mask_val = full_slate_mask[val_idx] if len(val_idx) > 0 else full_slate_mask_train
    y_route_train = y_route[train_idx]
    y_route_val = y_route[val_idx] if len(val_idx) > 0 else y_route_train
    state_value_target_train = state_value_target[train_idx]
    state_value_target_val = state_value_target[val_idx] if len(val_idx) > 0 else state_value_target_train
    pairwise_pairs_train = [pairwise_pairs[int(i)] for i in train_idx.tolist()]
    pairwise_pairs_val = [pairwise_pairs[int(i)] for i in val_idx.tolist()] if len(val_idx) > 0 else pairwise_pairs_train
    preview_route_targets_train = [preview_route_targets[int(i)] for i in train_idx.tolist()]
    preview_route_targets_val = [preview_route_targets[int(i)] for i in val_idx.tolist()] if len(val_idx) > 0 else preview_route_targets_train

    mean, std = _compute_feature_stats(x_train)
    x_train_n = _normalize_inputs(x_train, mean, std)
    x_val_n = _normalize_inputs(x_val, mean, std)
    if bool(path_valid_mask_train.any().item()):
        path_train_flat = path_x_train[path_valid_mask_train]
        path_mean, path_std = _compute_feature_stats(path_train_flat)
    else:
        path_mean = torch.zeros((len(path_feature_names),), dtype=torch.float32)
        path_std = torch.ones((len(path_feature_names),), dtype=torch.float32)
    path_x_train_n = _normalize_inputs(path_x_train, path_mean.reshape(1, 1, -1), path_std.reshape(1, 1, -1))
    path_x_val_n = _normalize_inputs(path_x_val, path_mean.reshape(1, 1, -1), path_std.reshape(1, 1, -1))
    if bool(preview_mask_train.any().item()):
        preview_train_flat = preview_x_train[preview_mask_train]
        preview_mean, preview_std = _compute_feature_stats(preview_train_flat)
    else:
        preview_mean = torch.zeros((len(preview_feature_names),), dtype=torch.float32)
        preview_std = torch.ones((len(preview_feature_names),), dtype=torch.float32)
    preview_x_train_n = _normalize_inputs(preview_x_train, preview_mean.reshape(1, 1, -1), preview_std.reshape(1, 1, -1))
    preview_x_val_n = _normalize_inputs(preview_x_val, preview_mean.reshape(1, 1, -1), preview_std.reshape(1, 1, -1))
    if bool(provenance_mask_train.any().item()):
        provenance_train_flat = provenance_x_train[provenance_mask_train]
        provenance_mean, provenance_std = _compute_feature_stats(provenance_train_flat)
    else:
        provenance_mean = torch.zeros((len(provenance_feature_names),), dtype=torch.float32)
        provenance_std = torch.ones((len(provenance_feature_names),), dtype=torch.float32)
    provenance_x_train_n = _normalize_inputs(
        provenance_x_train,
        provenance_mean.reshape(1, 1, 1, -1),
        provenance_std.reshape(1, 1, 1, -1),
    )
    provenance_x_val_n = _normalize_inputs(
        provenance_x_val,
        provenance_mean.reshape(1, 1, 1, -1),
        provenance_std.reshape(1, 1, 1, -1),
    )

    model = _RepairControllerSharedNet(
        int(x.shape[1]),
        int(hidden_dim),
        n_macro_actions=len(macro_action_names),
        n_routes=len(route_names),
        path_input_dim=len(path_feature_names),
        preview_input_dim=len(preview_feature_names),
        provenance_input_dim=len(provenance_feature_names),
    ).to(dtype=torch.float32)
    _maybe_init_model_from_bundle(model, init_bundle)
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))

    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_val = float("inf")
    patience = max(20, int(0.20 * max(1, int(epochs))))
    bad_epochs = 0

    def _loss_fn(
        outputs: dict[str, torch.Tensor],
        target_preview_utility: torch.Tensor,
        target_preview_regret: torch.Tensor,
        target_preview_mask: torch.Tensor,
        target_path: torch.Tensor,
        target_path_mask: torch.Tensor,
        target_full_mask: torch.Tensor,
        target_pairs: Sequence[Sequence[tuple[int, int, float]]],
        target_route: torch.Tensor,
        target_state_value: torch.Tensor,
        target_path_valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        preview_scores = outputs["preview_utility"]
        loss = preview_scores.sum() * 0.0
        listwise_rows = target_full_mask & (target_preview_mask.sum(dim=-1) >= 2)
        if bool(listwise_rows.any().item()):
            loss = loss + _listwise_slate_loss(
                preview_scores[listwise_rows],
                target_preview_utility[listwise_rows],
                target_preview_mask[listwise_rows],
                beta=float(listwise_beta),
                min_gap=float(score_gap_floor),
            )
        partial_rows = (~target_full_mask) & (target_preview_mask.sum(dim=-1) >= 2)
        partial_scores = preview_scores[partial_rows]
        partial_pairs = [target_pairs[idx] for idx, keep in enumerate(partial_rows.tolist()) if bool(keep)]
        if partial_scores.shape[0] > 0:
            loss = loss + _pairwise_rank_loss(
                partial_scores,
                partial_pairs,
                gap_scale=float(pairwise_gap_scale),
            )
        if float(path_ce_weight) > 0.0 and bool(target_path_mask.any().item()):
            loss = loss + float(path_ce_weight) * _masked_path_cross_entropy(
                outputs["path_logits"][target_path_mask],
                target_path[target_path_mask],
                target_path_valid_mask[target_path_mask],
            )
        if float(route_aux_weight) > 0.0:
            loss = loss + float(route_aux_weight) * F.cross_entropy(outputs["route_logits"], target_route)
        if float(preview_regret_weight) > 0.0 and bool(target_preview_mask.any().item()):
            loss = loss + float(preview_regret_weight) * F.smooth_l1_loss(
                outputs["preview_regret"][target_preview_mask],
                target_preview_regret[target_preview_mask],
            )
        if float(state_value_weight) > 0.0:
            state_mask = target_preview_mask.any(dim=-1)
            loss = loss + float(state_value_weight) * F.smooth_l1_loss(
                outputs["value_pred"][state_mask],
                target_state_value[state_mask],
            )
        return loss

    for _epoch in range(max(1, int(epochs))):
        model.train()
        outputs = model(
            x_train_n,
            path_x_train_n,
            path_valid_mask_train,
            preview_x_train_n,
            preview_path_indices_train,
            preview_mask_train,
            provenance_x_train_n,
            provenance_mask_train,
        )
        loss = _loss_fn(
            outputs,
            y_preview_utility_train,
            y_preview_regret_train,
            preview_mask_train,
            y_path_train,
            path_label_mask_train,
            full_slate_mask_train,
            pairwise_pairs_train,
            y_route_train,
            state_value_target_train,
            path_valid_mask_train,
        )
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        opt.step()

        model.eval()
        with torch.no_grad():
            outputs_val = model(
                x_val_n,
                path_x_val_n,
                path_valid_mask_val,
                preview_x_val_n,
                preview_path_indices_val,
                preview_mask_val,
                provenance_x_val_n,
                provenance_mask_val,
            )
            val_loss = _loss_fn(
                outputs_val,
                y_preview_utility_val,
                y_preview_regret_val,
                preview_mask_val,
                y_path_val,
                path_label_mask_val,
                full_slate_mask_val,
                pairwise_pairs_val,
                y_route_val,
                state_value_target_val,
                path_valid_mask_val,
            )
            val_loss_f = float(val_loss.item())
        if val_loss_f + 1.0e-8 < best_val:
            best_val = val_loss_f
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        train_out = model(
            x_train_n,
            path_x_train_n,
            path_valid_mask_train,
            preview_x_train_n,
            preview_path_indices_train,
            preview_mask_train,
            provenance_x_train_n,
            provenance_mask_train,
        )
        val_out = model(
            x_val_n,
            path_x_val_n,
            path_valid_mask_val,
            preview_x_val_n,
            preview_path_indices_val,
            preview_mask_val,
            provenance_x_val_n,
            provenance_mask_val,
        )
        train_metrics = _slate_rank_metrics(
            train_out["preview_utility"],
            y_preview_utility_train,
            preview_mask_train,
            pairwise_pairs_train,
            prefix="candidate",
        )
        val_metrics = _slate_rank_metrics(
            val_out["preview_utility"],
            y_preview_utility_val,
            preview_mask_val,
            pairwise_pairs_val,
            prefix="candidate",
        )
        train_metrics.update(_route_emergence_metrics(
            train_out["preview_utility"],
            y_preview_utility_train,
            preview_mask_train,
            preview_route_targets_train,
            prefix="candidate",
        ))
        val_metrics.update(_route_emergence_metrics(
            val_out["preview_utility"],
            y_preview_utility_val,
            preview_mask_val,
            preview_route_targets_val,
            prefix="candidate",
        ))
        if bool(path_label_mask_train.any().item()):
            train_metrics.update(_path_metrics_from_logits(
                train_out["path_logits"][path_label_mask_train],
                y_path_train[path_label_mask_train],
                path_valid_mask_train[path_label_mask_train],
            ))
        if bool(path_label_mask_val.any().item()):
            val_metrics.update(_path_metrics_from_logits(
                val_out["path_logits"][path_label_mask_val],
                y_path_val[path_label_mask_val],
                path_valid_mask_val[path_label_mask_val],
            ))
        if float(route_aux_weight) > 0.0:
            train_metrics.update(_flat_classification_metrics(
                train_out["route_logits"], y_route_train, prefix="route_aux"
            ))
            val_metrics.update(_flat_classification_metrics(
                val_out["route_logits"], y_route_val, prefix="route_aux"
            ))
        if bool(preview_mask_train.any().item()):
            train_metrics.update(_regression_metrics(
                train_out["preview_regret"][preview_mask_train],
                y_preview_regret_train[preview_mask_train],
                prefix="preview_regret",
            ))
        if bool(preview_mask_val.any().item()):
            val_metrics.update(_regression_metrics(
                val_out["preview_regret"][preview_mask_val],
                y_preview_regret_val[preview_mask_val],
                prefix="preview_regret",
            ))
        state_mask_train = preview_mask_train.any(dim=-1)
        state_mask_val = preview_mask_val.any(dim=-1)
        if bool(state_mask_train.any().item()):
            train_metrics.update(_regression_metrics(
                train_out["value_pred"][state_mask_train],
                state_value_target_train[state_mask_train],
                prefix="state_value",
            ))
        if bool(state_mask_val.any().item()):
            val_metrics.update(_regression_metrics(
                val_out["value_pred"][state_mask_val],
                state_value_target_val[state_mask_val],
                prefix="state_value",
            ))

    return {
        "model_kind": REPAIR_CRITIC_SHARED_MODEL_KIND,
        "feature_names": feature_names,
        "path_feature_names": path_feature_names,
        "preview_feature_names": preview_feature_names,
        "provenance_feature_names": provenance_feature_names,
        "head_names": head_names,
        "macro_action_names": macro_action_names,
        "route_names": route_names,
        "mode_names": mode_names,
        "path_relation_names": relation_names,
        "hidden_dim": int(hidden_dim),
        "reward_per_s_scale": float(init_bundle.get("reward_per_s_scale", 1.0)) if isinstance(init_bundle, dict) else 1.0,
        "feature_mean": mean.detach().cpu(),
        "feature_std": std.detach().cpu(),
        "path_feature_mean": torch.as_tensor(path_mean.detach().cpu(), dtype=torch.float32),
        "path_feature_std": torch.as_tensor(path_std.detach().cpu(), dtype=torch.float32),
        "preview_feature_mean": torch.as_tensor(preview_mean.detach().cpu(), dtype=torch.float32),
        "preview_feature_std": torch.as_tensor(preview_std.detach().cpu(), dtype=torch.float32),
        "provenance_feature_mean": torch.as_tensor(provenance_mean.detach().cpu(), dtype=torch.float32),
        "provenance_feature_std": torch.as_tensor(provenance_std.detach().cpu(), dtype=torch.float32),
        "utility_weights": dict(REPAIR_CRITIC_DEFAULT_UTILITY_WEIGHTS),
        "unified_candidate_ranker_trained": True,
        "unified_candidate_ranker_target": "same_parent_best_exact_candidate",
        "unified_candidate_route_tau": 1.0,
        "unified_candidate_action_names": sorted(name for name in observed_action_names if name),
        "repair_tuple_ranker_trained": False,
        "repair_slate_ranker_trained": False,
        "macro_head_trained": False,
        "route_head_trained": float(route_aux_weight) > 0.0,
        "q_head_trained": False,
        "value_head_trained": True,
        "regret_head_trained": True,
        "path_head_trained": bool(path_ce_weight > 0.0),
        "path_action_head_trained": False,
        "path_relation_head_trained": False,
        "path_mode_head_trained": False,
        "path_improve_head_trained": False,
        "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "metrics": {
            "n_examples": int(len(training_rows)),
            "n_full_slate_examples": int(full_slate_mask.sum().item()),
            "n_pairwise_examples": int((~full_slate_mask).sum().item()),
            "n_pairwise_pairs": int(sum(len(pairs) for pairs in pairwise_pairs)),
            "n_path_examples": int(path_label_mask.sum().item()),
            "n_tuple_examples": int(preview_mask.sum().item()),
            "n_provenance_tuples": int(provenance_mask.sum().item()),
            "n_train": int(x_train.shape[0]),
            "n_val": int(x_val.shape[0]),
            "best_val_loss": float(best_val),
            "reward_source_counts": dict(reward_source_counts),
            "train": train_metrics,
            "val": val_metrics,
        },
    }


def train_shared_candidate_dual_ranker(
    rows: Sequence[Any],
    *,
    hidden_dim: int = 32,
    epochs: int = 120,
    lr: float = 5.0e-3,
    weight_decay: float = 1.0e-4,
    val_fraction: float = 0.2,
    seed: int = 0,
    listwise_beta: float = 2.0,
    score_gap_floor: float = 1.0e-3,
    pairwise_gap_scale: float = 0.1,
    pairwise_weight: float = 0.25,
    repair_rank_weight: float = 1.0,
    build_rank_weight: float = 1.0,
    common_rank_weight: float = 1.0,
    path_ce_weight: float = 0.05,
    common_value_weight: float = 0.25,
    common_regret_weight: float = 1.0,
    common_q_weight: float = 0.20,
    preview_value_weight: float = 0.10,
    preview_regret_weight: float = 0.10,
    common_state_value_weight: float = 0.10,
    repair_state_value_weight: float = 0.10,
    build_state_value_weight: float = 0.10,
    oracle_path_weight: float = 0.05,
    oracle_relation_weight: float = 0.05,
    oracle_mode_weight: float = 0.05,
    oracle_truth_weight: float = 0.05,
    oracle_mode_best_weight: float = 0.05,
    oracle_rank_weight: float = 0.05,
    oracle_stability_weight: float = 0.05,
    oracle_coverage_weight: float = 0.05,
    init_bundle: dict[str, Any] | None = None,
) -> dict[str, Any]:
    torch.manual_seed(int(seed))
    training_rows, reward_source_counts = _build_unified_candidate_rows(
        rows,
        score_gap_floor=float(score_gap_floor),
        common_value_weight=float(common_value_weight),
        common_regret_weight=float(common_regret_weight),
    )
    if len(training_rows) < 8:
        raise ValueError("Need at least 8 same-parent mixed candidate slates to train a shared candidate dual ranker.")

    feature_names = list(REPAIR_CRITIC_FEATURE_NAMES)
    path_feature_names = list(REPAIR_CRITIC_PATH_FEATURE_NAMES)
    preview_feature_names = list(UNIFIED_CANDIDATE_PREVIEW_FEATURE_NAMES)
    provenance_feature_names = list(preview_feature_names)
    n_rows = len(training_rows)
    x = torch.tensor(
        [[float(row["features"].get(name, 0.0)) for name in feature_names] for row in training_rows],
        dtype=torch.float32,
    )
    max_paths = max((len(tuple(row.get("path_rows", ()))) for row in training_rows), default=0)
    max_paths = max(1, int(max_paths))
    max_preview = max((len(list(row.get("preview_rows", []) or [])) for row in training_rows), default=0)
    max_preview = max(1, int(max_preview))
    max_provenance = max(
        (
            len(list(preview_row.get("provenance_rows", []) or [preview_row]))
            for row in training_rows
            for preview_row in list(row.get("preview_rows", []) or [])
            if isinstance(preview_row, Mapping)
        ),
        default=0,
    )
    max_provenance = max(1, int(max_provenance))

    path_x = torch.zeros((n_rows, max_paths, len(path_feature_names)), dtype=torch.float32)
    path_valid_mask = torch.zeros((n_rows, max_paths), dtype=torch.bool)
    y_repair_path = torch.full((n_rows,), -100, dtype=torch.long)
    repair_path_mask = torch.zeros((n_rows,), dtype=torch.bool)
    y_oracle_path = torch.full((n_rows,), -100, dtype=torch.long)
    oracle_path_mask = torch.zeros((n_rows,), dtype=torch.bool)
    y_oracle_relation = torch.full((n_rows, max_paths), -100, dtype=torch.long)
    oracle_relation_mask = torch.zeros((n_rows, max_paths), dtype=torch.bool)
    y_oracle_mode = torch.full((n_rows, max_paths), -100, dtype=torch.long)
    oracle_mode_mask = torch.zeros((n_rows, max_paths), dtype=torch.bool)
    preview_x = torch.zeros((n_rows, max_preview, len(preview_feature_names)), dtype=torch.float32)
    preview_mask = torch.zeros((n_rows, max_preview), dtype=torch.bool)
    preview_path_indices = torch.zeros((n_rows, max_preview), dtype=torch.long)
    provenance_x = torch.zeros((n_rows, max_preview, max_provenance, len(provenance_feature_names)), dtype=torch.float32)
    provenance_mask = torch.zeros((n_rows, max_preview, max_provenance), dtype=torch.bool)
    y_preview_utility = torch.zeros((n_rows, max_preview), dtype=torch.float32)
    y_preview_value = torch.zeros((n_rows, max_preview), dtype=torch.float32)
    y_preview_regret = torch.zeros((n_rows, max_preview), dtype=torch.float32)
    y_preview_q = torch.zeros((n_rows, max_preview), dtype=torch.float32)
    preview_value_mask = torch.zeros((n_rows, max_preview), dtype=torch.bool)
    repair_preview_mask = torch.zeros((n_rows, max_preview), dtype=torch.bool)
    build_preview_mask = torch.zeros((n_rows, max_preview), dtype=torch.bool)
    y_common_state_value = torch.zeros((n_rows,), dtype=torch.float32)
    y_repair_state_value = torch.zeros((n_rows,), dtype=torch.float32)
    y_build_state_value = torch.zeros((n_rows,), dtype=torch.float32)
    repair_state_value_mask = torch.zeros((n_rows,), dtype=torch.bool)
    build_state_value_mask = torch.zeros((n_rows,), dtype=torch.bool)
    y_oracle_truth = torch.zeros((n_rows, max_preview), dtype=torch.float32)
    oracle_truth_mask = torch.zeros((n_rows, max_preview), dtype=torch.bool)
    y_oracle_mode_best = torch.zeros((n_rows, max_preview), dtype=torch.float32)
    oracle_mode_best_mask = torch.zeros((n_rows, max_preview), dtype=torch.bool)
    y_oracle_rank = torch.zeros((n_rows, max_preview), dtype=torch.float32)
    oracle_rank_mask = torch.zeros((n_rows, max_preview), dtype=torch.bool)
    y_oracle_stability = torch.zeros((n_rows, max_preview), dtype=torch.float32)
    oracle_stability_mask = torch.zeros((n_rows, max_preview), dtype=torch.bool)
    y_oracle_coverage = torch.zeros((n_rows,), dtype=torch.float32)
    oracle_coverage_mask = torch.zeros((n_rows,), dtype=torch.bool)
    repair_action_names_seen: set[str] = set()
    build_action_names_seen: set[str] = set()

    for i, row in enumerate(training_rows):
        path_rows = tuple(row.get("path_rows", ()) or ())
        for j, path_row in enumerate(path_rows[:max_paths]):
            path_valid_mask[i, j] = True
            path_x[i, j] = torch.tensor(
                repair_path_feature_vector(path_row, feature_names=path_feature_names),
                dtype=torch.float32,
            )
        repair_path_target_index = row.get("repair_path_target_index", None)
        if isinstance(repair_path_target_index, int) and 0 <= repair_path_target_index < min(len(path_rows), max_paths):
            y_repair_path[i] = int(repair_path_target_index)
            repair_path_mask[i] = True
        oracle_path_target_index = row.get("oracle_path_target_index", None)
        if isinstance(oracle_path_target_index, int) and 0 <= oracle_path_target_index < min(len(path_rows), max_paths):
            y_oracle_path[i] = int(oracle_path_target_index)
            oracle_path_mask[i] = True
        oracle_relation_targets = list(row.get("oracle_relation_targets", []) or [])
        for j in range(min(max_paths, len(oracle_relation_targets))):
            target = oracle_relation_targets[j]
            if isinstance(target, int) and target >= 0:
                y_oracle_relation[i, j] = int(target)
                oracle_relation_mask[i, j] = True
        oracle_mode_targets = list(row.get("oracle_mode_targets", []) or [])
        for j in range(min(max_paths, len(oracle_mode_targets))):
            target = oracle_mode_targets[j]
            if isinstance(target, int) and target >= 0:
                y_oracle_mode[i, j] = int(target)
                oracle_mode_mask[i, j] = True
        y_common_state_value[i] = float(row.get("common_state_value_target", row.get("state_value_target", 0.0)) or 0.0)
        y_repair_state_value[i] = float(row.get("repair_state_value_target", 0.0) or 0.0)
        repair_state_value_mask[i] = bool(row.get("repair_state_value_mask", False))
        y_build_state_value[i] = float(row.get("build_state_value_target", 0.0) or 0.0)
        build_state_value_mask[i] = bool(row.get("build_state_value_mask", False))
        if bool(row.get("oracle_truth_in_slate_mask", False)):
            oracle_coverage_mask[i] = True
            y_oracle_coverage[i] = 1.0 if bool(row.get("oracle_truth_in_slate", False)) else 0.0
        preview_rows = list(row.get("preview_rows", []) or [])
        preview_path_idx_list = list(row.get("preview_path_indices", []) or [])
        preview_utilities = list(row.get("preview_utility_targets", []) or [])
        preview_values = list(row.get("preview_value_targets", []) or [])
        preview_regrets = list(row.get("preview_regret_targets", []) or [])
        preview_q_targets = list(row.get("preview_q_targets", []) or [])
        preview_value_flags = list(row.get("preview_value_mask", []) or [])
        preview_routes = list(row.get("preview_route_targets", []) or [])
        preview_oracle_truth_targets = list(row.get("preview_oracle_truth_targets", []) or [])
        preview_oracle_truth_flags = list(row.get("preview_oracle_truth_mask", []) or [])
        preview_oracle_mode_targets = list(row.get("preview_oracle_mode_best_targets", []) or [])
        preview_oracle_mode_flags = list(row.get("preview_oracle_mode_best_mask", []) or [])
        preview_oracle_rank_targets = list(row.get("preview_oracle_rank_targets", []) or [])
        preview_oracle_rank_flags = list(row.get("preview_oracle_rank_mask", []) or [])
        preview_oracle_stability_targets = list(row.get("preview_oracle_stability_targets", []) or [])
        preview_oracle_stability_flags = list(row.get("preview_oracle_stability_mask", []) or [])
        for j in range(min(max_preview, len(preview_rows), len(preview_path_idx_list), len(preview_utilities))):
            preview_mask[i, j] = True
            preview_x[i, j] = torch.tensor(
                unified_candidate_preview_feature_vector(preview_rows[j], feature_names=preview_feature_names),
                dtype=torch.float32,
            )
            preview_path_indices[i, j] = int(max(0, min(max_paths - 1, int(preview_path_idx_list[j]))))
            provenance_rows = list(preview_rows[j].get("provenance_rows", []) or [preview_rows[j]])
            for k in range(min(max_provenance, len(provenance_rows))):
                provenance_mask[i, j, k] = True
                provenance_x[i, j, k] = torch.tensor(
                    unified_candidate_preview_feature_vector(provenance_rows[k], feature_names=provenance_feature_names),
                    dtype=torch.float32,
                )
            y_preview_utility[i, j] = float(preview_utilities[j])
            if j < len(preview_values):
                y_preview_value[i, j] = float(preview_values[j])
            if j < len(preview_regrets):
                y_preview_regret[i, j] = max(0.0, float(preview_regrets[j]))
            if j < len(preview_q_targets):
                y_preview_q[i, j] = float(preview_q_targets[j])
            if j < len(preview_value_flags):
                preview_value_mask[i, j] = bool(preview_value_flags[j])
            if j < len(preview_oracle_truth_flags) and bool(preview_oracle_truth_flags[j]):
                oracle_truth_mask[i, j] = True
                y_oracle_truth[i, j] = 1.0 if j < len(preview_oracle_truth_targets) and bool(preview_oracle_truth_targets[j]) else 0.0
            if j < len(preview_oracle_mode_flags) and bool(preview_oracle_mode_flags[j]):
                oracle_mode_best_mask[i, j] = True
                y_oracle_mode_best[i, j] = 1.0 if j < len(preview_oracle_mode_targets) and bool(preview_oracle_mode_targets[j]) else 0.0
            if j < len(preview_oracle_rank_flags) and bool(preview_oracle_rank_flags[j]):
                oracle_rank_mask[i, j] = True
                y_oracle_rank[i, j] = float(preview_oracle_rank_targets[j]) if j < len(preview_oracle_rank_targets) else 0.0
            if j < len(preview_oracle_stability_flags) and bool(preview_oracle_stability_flags[j]):
                oracle_stability_mask[i, j] = True
                y_oracle_stability[i, j] = 1.0 if j < len(preview_oracle_stability_targets) and bool(preview_oracle_stability_targets[j]) else 0.0
            route_name = str(preview_routes[j] if j < len(preview_routes) else preview_rows[j].get("route_source", "")).strip().lower()
            if route_name == "repair":
                repair_preview_mask[i, j] = True
                repair_action_names_seen.add(_normalize_action_name(preview_rows[j].get("action", "")))
            elif route_name == "build":
                build_preview_mask[i, j] = True
                build_action_names_seen.add(_normalize_action_name(preview_rows[j].get("action", "")))

    repair_pairwise_pairs = _masked_pairwise_pairs_from_targets(
        y_preview_utility,
        repair_preview_mask,
        gap_floor=float(score_gap_floor),
    )
    build_pairwise_pairs = _masked_pairwise_pairs_from_targets(
        y_preview_utility,
        build_preview_mask,
        gap_floor=float(score_gap_floor),
    )
    common_pairwise_pairs = _masked_pairwise_pairs_from_targets(
        y_preview_q,
        preview_mask,
        gap_floor=float(score_gap_floor),
    )

    train_idx, val_idx = _split_indices(n_rows, float(val_fraction), int(seed))
    x_train = x[train_idx]
    x_val = x[val_idx] if len(val_idx) > 0 else x_train
    path_x_train = path_x[train_idx]
    path_x_val = path_x[val_idx] if len(val_idx) > 0 else path_x_train
    path_valid_mask_train = path_valid_mask[train_idx]
    path_valid_mask_val = path_valid_mask[val_idx] if len(val_idx) > 0 else path_valid_mask_train
    y_repair_path_train = y_repair_path[train_idx]
    y_repair_path_val = y_repair_path[val_idx] if len(val_idx) > 0 else y_repair_path_train
    repair_path_mask_train = repair_path_mask[train_idx]
    repair_path_mask_val = repair_path_mask[val_idx] if len(val_idx) > 0 else repair_path_mask_train
    y_oracle_path_train = y_oracle_path[train_idx]
    y_oracle_path_val = y_oracle_path[val_idx] if len(val_idx) > 0 else y_oracle_path_train
    oracle_path_mask_train = oracle_path_mask[train_idx]
    oracle_path_mask_val = oracle_path_mask[val_idx] if len(val_idx) > 0 else oracle_path_mask_train
    y_oracle_relation_train = y_oracle_relation[train_idx]
    y_oracle_relation_val = y_oracle_relation[val_idx] if len(val_idx) > 0 else y_oracle_relation_train
    oracle_relation_mask_train = oracle_relation_mask[train_idx]
    oracle_relation_mask_val = oracle_relation_mask[val_idx] if len(val_idx) > 0 else oracle_relation_mask_train
    y_oracle_mode_train = y_oracle_mode[train_idx]
    y_oracle_mode_val = y_oracle_mode[val_idx] if len(val_idx) > 0 else y_oracle_mode_train
    oracle_mode_mask_train = oracle_mode_mask[train_idx]
    oracle_mode_mask_val = oracle_mode_mask[val_idx] if len(val_idx) > 0 else oracle_mode_mask_train
    preview_x_train = preview_x[train_idx]
    preview_x_val = preview_x[val_idx] if len(val_idx) > 0 else preview_x_train
    preview_mask_train = preview_mask[train_idx]
    preview_mask_val = preview_mask[val_idx] if len(val_idx) > 0 else preview_mask_train
    preview_path_indices_train = preview_path_indices[train_idx]
    preview_path_indices_val = preview_path_indices[val_idx] if len(val_idx) > 0 else preview_path_indices_train
    provenance_x_train = provenance_x[train_idx]
    provenance_x_val = provenance_x[val_idx] if len(val_idx) > 0 else provenance_x_train
    provenance_mask_train = provenance_mask[train_idx]
    provenance_mask_val = provenance_mask[val_idx] if len(val_idx) > 0 else provenance_mask_train
    y_preview_utility_train = y_preview_utility[train_idx]
    y_preview_utility_val = y_preview_utility[val_idx] if len(val_idx) > 0 else y_preview_utility_train
    y_preview_value_train = y_preview_value[train_idx]
    y_preview_value_val = y_preview_value[val_idx] if len(val_idx) > 0 else y_preview_value_train
    y_preview_regret_train = y_preview_regret[train_idx]
    y_preview_regret_val = y_preview_regret[val_idx] if len(val_idx) > 0 else y_preview_regret_train
    y_preview_q_train = y_preview_q[train_idx]
    y_preview_q_val = y_preview_q[val_idx] if len(val_idx) > 0 else y_preview_q_train
    preview_value_mask_train = preview_value_mask[train_idx]
    preview_value_mask_val = preview_value_mask[val_idx] if len(val_idx) > 0 else preview_value_mask_train
    repair_preview_mask_train = repair_preview_mask[train_idx]
    repair_preview_mask_val = repair_preview_mask[val_idx] if len(val_idx) > 0 else repair_preview_mask_train
    build_preview_mask_train = build_preview_mask[train_idx]
    build_preview_mask_val = build_preview_mask[val_idx] if len(val_idx) > 0 else build_preview_mask_train
    y_common_state_value_train = y_common_state_value[train_idx]
    y_common_state_value_val = y_common_state_value[val_idx] if len(val_idx) > 0 else y_common_state_value_train
    y_repair_state_value_train = y_repair_state_value[train_idx]
    y_repair_state_value_val = y_repair_state_value[val_idx] if len(val_idx) > 0 else y_repair_state_value_train
    repair_state_value_mask_train = repair_state_value_mask[train_idx]
    repair_state_value_mask_val = repair_state_value_mask[val_idx] if len(val_idx) > 0 else repair_state_value_mask_train
    y_build_state_value_train = y_build_state_value[train_idx]
    y_build_state_value_val = y_build_state_value[val_idx] if len(val_idx) > 0 else y_build_state_value_train
    build_state_value_mask_train = build_state_value_mask[train_idx]
    build_state_value_mask_val = build_state_value_mask[val_idx] if len(val_idx) > 0 else build_state_value_mask_train
    y_oracle_truth_train = y_oracle_truth[train_idx]
    y_oracle_truth_val = y_oracle_truth[val_idx] if len(val_idx) > 0 else y_oracle_truth_train
    oracle_truth_mask_train = oracle_truth_mask[train_idx]
    oracle_truth_mask_val = oracle_truth_mask[val_idx] if len(val_idx) > 0 else oracle_truth_mask_train
    y_oracle_mode_best_train = y_oracle_mode_best[train_idx]
    y_oracle_mode_best_val = y_oracle_mode_best[val_idx] if len(val_idx) > 0 else y_oracle_mode_best_train
    oracle_mode_best_mask_train = oracle_mode_best_mask[train_idx]
    oracle_mode_best_mask_val = oracle_mode_best_mask[val_idx] if len(val_idx) > 0 else oracle_mode_best_mask_train
    y_oracle_rank_train = y_oracle_rank[train_idx]
    y_oracle_rank_val = y_oracle_rank[val_idx] if len(val_idx) > 0 else y_oracle_rank_train
    oracle_rank_mask_train = oracle_rank_mask[train_idx]
    oracle_rank_mask_val = oracle_rank_mask[val_idx] if len(val_idx) > 0 else oracle_rank_mask_train
    y_oracle_stability_train = y_oracle_stability[train_idx]
    y_oracle_stability_val = y_oracle_stability[val_idx] if len(val_idx) > 0 else y_oracle_stability_train
    oracle_stability_mask_train = oracle_stability_mask[train_idx]
    oracle_stability_mask_val = oracle_stability_mask[val_idx] if len(val_idx) > 0 else oracle_stability_mask_train
    y_oracle_coverage_train = y_oracle_coverage[train_idx]
    y_oracle_coverage_val = y_oracle_coverage[val_idx] if len(val_idx) > 0 else y_oracle_coverage_train
    oracle_coverage_mask_train = oracle_coverage_mask[train_idx]
    oracle_coverage_mask_val = oracle_coverage_mask[val_idx] if len(val_idx) > 0 else oracle_coverage_mask_train
    repair_pairwise_pairs_train = [repair_pairwise_pairs[int(i)] for i in train_idx.tolist()]
    repair_pairwise_pairs_val = [repair_pairwise_pairs[int(i)] for i in val_idx.tolist()] if len(val_idx) > 0 else repair_pairwise_pairs_train
    build_pairwise_pairs_train = [build_pairwise_pairs[int(i)] for i in train_idx.tolist()]
    build_pairwise_pairs_val = [build_pairwise_pairs[int(i)] for i in val_idx.tolist()] if len(val_idx) > 0 else build_pairwise_pairs_train
    common_pairwise_pairs_train = [common_pairwise_pairs[int(i)] for i in train_idx.tolist()]
    common_pairwise_pairs_val = [common_pairwise_pairs[int(i)] for i in val_idx.tolist()] if len(val_idx) > 0 else common_pairwise_pairs_train

    mean, std = _compute_feature_stats(x_train)
    x_train_n = _normalize_inputs(x_train, mean, std)
    x_val_n = _normalize_inputs(x_val, mean, std)
    if bool(path_valid_mask_train.any().item()):
        path_train_flat = path_x_train[path_valid_mask_train]
        path_mean, path_std = _compute_feature_stats(path_train_flat)
    else:
        path_mean = torch.zeros((len(path_feature_names),), dtype=torch.float32)
        path_std = torch.ones((len(path_feature_names),), dtype=torch.float32)
    path_x_train_n = _normalize_inputs(path_x_train, path_mean.reshape(1, 1, -1), path_std.reshape(1, 1, -1))
    path_x_val_n = _normalize_inputs(path_x_val, path_mean.reshape(1, 1, -1), path_std.reshape(1, 1, -1))
    if bool(preview_mask_train.any().item()):
        preview_train_flat = preview_x_train[preview_mask_train]
        preview_mean, preview_std = _compute_feature_stats(preview_train_flat)
    else:
        preview_mean = torch.zeros((len(preview_feature_names),), dtype=torch.float32)
        preview_std = torch.ones((len(preview_feature_names),), dtype=torch.float32)
    preview_x_train_n = _normalize_inputs(preview_x_train, preview_mean.reshape(1, 1, -1), preview_std.reshape(1, 1, -1))
    preview_x_val_n = _normalize_inputs(preview_x_val, preview_mean.reshape(1, 1, -1), preview_std.reshape(1, 1, -1))
    if bool(provenance_mask_train.any().item()):
        provenance_train_flat = provenance_x_train[provenance_mask_train]
        provenance_mean, provenance_std = _compute_feature_stats(provenance_train_flat)
    else:
        provenance_mean = torch.zeros((len(provenance_feature_names),), dtype=torch.float32)
        provenance_std = torch.ones((len(provenance_feature_names),), dtype=torch.float32)
    provenance_x_train_n = _normalize_inputs(
        provenance_x_train,
        provenance_mean.reshape(1, 1, 1, -1),
        provenance_std.reshape(1, 1, 1, -1),
    )
    provenance_x_val_n = _normalize_inputs(
        provenance_x_val,
        provenance_mean.reshape(1, 1, 1, -1),
        provenance_std.reshape(1, 1, 1, -1),
    )

    repair_state_mean = float(y_repair_state_value_train[repair_state_value_mask_train].mean().item()) if bool(repair_state_value_mask_train.any().item()) else 0.0
    repair_state_std = float(y_repair_state_value_train[repair_state_value_mask_train].std(unbiased=False).item()) if bool(repair_state_value_mask_train.any().item()) else 1.0
    repair_state_std = max(1.0e-6, repair_state_std)
    build_state_mean = float(y_build_state_value_train[build_state_value_mask_train].mean().item()) if bool(build_state_value_mask_train.any().item()) else 0.0
    build_state_std = float(y_build_state_value_train[build_state_value_mask_train].std(unbiased=False).item()) if bool(build_state_value_mask_train.any().item()) else 1.0
    build_state_std = max(1.0e-6, build_state_std)
    common_state_mean = float(y_common_state_value_train.mean().item()) if int(y_common_state_value_train.numel()) > 0 else 0.0
    common_state_std = float(y_common_state_value_train.std(unbiased=False).item()) if int(y_common_state_value_train.numel()) > 1 else 1.0
    common_state_std = max(1.0e-6, common_state_std)
    y_repair_state_value_train_n = (y_repair_state_value_train - repair_state_mean) / repair_state_std
    y_repair_state_value_val_n = (y_repair_state_value_val - repair_state_mean) / repair_state_std
    y_build_state_value_train_n = (y_build_state_value_train - build_state_mean) / build_state_std
    y_build_state_value_val_n = (y_build_state_value_val - build_state_mean) / build_state_std
    y_common_state_value_train_n = (y_common_state_value_train - common_state_mean) / common_state_std
    y_common_state_value_val_n = (y_common_state_value_val - common_state_mean) / common_state_std

    model = _SharedCandidateDualRankerNet(
        int(x.shape[1]),
        int(hidden_dim),
        path_input_dim=len(path_feature_names),
        preview_input_dim=len(preview_feature_names),
        provenance_input_dim=len(provenance_feature_names),
    ).to(dtype=torch.float32)
    _maybe_init_model_from_bundle(model, init_bundle)
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))

    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_val = float("inf")
    patience = max(20, int(0.20 * max(1, int(epochs))))
    bad_epochs = 0

    def _route_loss(
        scores: torch.Tensor,
        route_mask: torch.Tensor,
        utility_targets: torch.Tensor,
        pairwise_pairs_local: Sequence[Sequence[tuple[int, int, float]]],
    ) -> torch.Tensor:
        loss_local = scores.sum() * 0.0
        informative_rows = route_mask.sum(dim=-1) >= 2
        if bool(informative_rows.any().item()):
            loss_local = loss_local + _listwise_slate_loss(
                scores[informative_rows],
                utility_targets[informative_rows],
                route_mask[informative_rows],
                beta=float(listwise_beta),
                min_gap=float(score_gap_floor),
            )
        if float(pairwise_weight) > 0.0:
            loss_local = loss_local + float(pairwise_weight) * _pairwise_rank_loss(
                scores,
                pairwise_pairs_local,
                gap_scale=float(pairwise_gap_scale),
            )
        return loss_local

    def _loss_fn(
        outputs: dict[str, torch.Tensor],
        utility_targets: torch.Tensor,
        repair_mask_local: torch.Tensor,
        build_mask_local: torch.Tensor,
        repair_pairs_local: Sequence[Sequence[tuple[int, int, float]]],
        build_pairs_local: Sequence[Sequence[tuple[int, int, float]]],
        repair_path_targets_local: torch.Tensor,
        repair_path_mask_local: torch.Tensor,
        oracle_path_targets_local: torch.Tensor,
        oracle_path_mask_local: torch.Tensor,
        oracle_relation_targets_local: torch.Tensor,
        oracle_relation_mask_local: torch.Tensor,
        oracle_mode_targets_local: torch.Tensor,
        oracle_mode_mask_local: torch.Tensor,
        path_valid_mask_local: torch.Tensor,
        preview_mask_local: torch.Tensor,
        repair_value_targets_local: torch.Tensor,
        repair_value_mask_local: torch.Tensor,
        build_value_targets_local: torch.Tensor,
        build_value_mask_local: torch.Tensor,
        common_value_targets_local: torch.Tensor,
        common_regret_targets_local: torch.Tensor,
        common_q_targets_local: torch.Tensor,
        common_preview_value_mask_local: torch.Tensor,
        common_pairs_local: Sequence[Sequence[tuple[int, int, float]]],
        common_state_value_targets_local: torch.Tensor,
        oracle_truth_targets_local: torch.Tensor,
        oracle_truth_mask_local: torch.Tensor,
        oracle_mode_best_targets_local: torch.Tensor,
        oracle_mode_best_mask_local: torch.Tensor,
        oracle_rank_targets_local: torch.Tensor,
        oracle_rank_mask_local: torch.Tensor,
        oracle_stability_targets_local: torch.Tensor,
        oracle_stability_mask_local: torch.Tensor,
        oracle_coverage_targets_local: torch.Tensor,
        oracle_coverage_mask_local: torch.Tensor,
    ) -> torch.Tensor:
        loss_local = outputs["repair_preview_score"].sum() * 0.0
        if float(repair_rank_weight) > 0.0:
            loss_local = loss_local + float(repair_rank_weight) * _route_loss(
                outputs["repair_preview_score"],
                repair_mask_local,
                utility_targets,
                repair_pairs_local,
            )
        if float(build_rank_weight) > 0.0:
            loss_local = loss_local + float(build_rank_weight) * _route_loss(
                outputs["build_preview_score"],
                build_mask_local,
                utility_targets,
                build_pairs_local,
            )
        if float(common_rank_weight) > 0.0:
            loss_local = loss_local + float(common_rank_weight) * _route_loss(
                outputs["common_preview_q"],
                preview_mask_local,
                common_q_targets_local,
                common_pairs_local,
            )
        if float(path_ce_weight) > 0.0 and bool(repair_path_mask_local.any().item()):
            loss_local = loss_local + float(path_ce_weight) * _masked_path_cross_entropy(
                outputs["path_logits"][repair_path_mask_local],
                repair_path_targets_local[repair_path_mask_local],
                path_valid_mask_local[repair_path_mask_local],
            )
        if float(oracle_path_weight) > 0.0 and bool(oracle_path_mask_local.any().item()):
            loss_local = loss_local + float(oracle_path_weight) * _masked_path_cross_entropy(
                outputs["path_logits"][oracle_path_mask_local],
                oracle_path_targets_local[oracle_path_mask_local],
                path_valid_mask_local[oracle_path_mask_local],
            )
        if float(oracle_relation_weight) > 0.0 and bool(oracle_relation_mask_local.any().item()):
            loss_local = loss_local + float(oracle_relation_weight) * F.cross_entropy(
                outputs["path_relation_logits"][oracle_relation_mask_local],
                oracle_relation_targets_local[oracle_relation_mask_local],
            )
        if float(oracle_mode_weight) > 0.0 and bool(oracle_mode_mask_local.any().item()):
            loss_local = loss_local + float(oracle_mode_weight) * F.cross_entropy(
                outputs["path_mode_logits"][oracle_mode_mask_local],
                oracle_mode_targets_local[oracle_mode_mask_local],
            )
        if float(preview_value_weight) > 0.0 and bool(common_preview_value_mask_local.any().item()):
            loss_local = loss_local + float(preview_value_weight) * F.smooth_l1_loss(
                outputs["preview_value"][common_preview_value_mask_local],
                common_value_targets_local[common_preview_value_mask_local],
            )
        if float(preview_regret_weight) > 0.0 and bool(preview_mask_local.any().item()):
            loss_local = loss_local + float(preview_regret_weight) * F.smooth_l1_loss(
                outputs["preview_regret"][preview_mask_local],
                common_regret_targets_local[preview_mask_local],
            )
        if float(common_q_weight) > 0.0 and bool(preview_mask_local.any().item()):
            loss_local = loss_local + float(common_q_weight) * F.smooth_l1_loss(
                outputs["common_preview_q"][preview_mask_local],
                common_q_targets_local[preview_mask_local],
            )
        if float(common_state_value_weight) > 0.0 and outputs["value_pred"].shape[0] > 0:
            loss_local = loss_local + float(common_state_value_weight) * F.smooth_l1_loss(
                outputs["value_pred"],
                common_state_value_targets_local,
            )
        if float(repair_state_value_weight) > 0.0 and bool(repair_value_mask_local.any().item()):
            loss_local = loss_local + float(repair_state_value_weight) * F.smooth_l1_loss(
                outputs["repair_state_value"][repair_value_mask_local],
                repair_value_targets_local[repair_value_mask_local],
            )
        if float(build_state_value_weight) > 0.0 and bool(build_value_mask_local.any().item()):
            loss_local = loss_local + float(build_state_value_weight) * F.smooth_l1_loss(
                outputs["build_state_value"][build_value_mask_local],
                build_value_targets_local[build_value_mask_local],
            )
        if float(oracle_truth_weight) > 0.0 and bool(oracle_truth_mask_local.any().item()):
            loss_local = loss_local + float(oracle_truth_weight) * F.binary_cross_entropy_with_logits(
                outputs["oracle_preview_truth_logit"][oracle_truth_mask_local],
                oracle_truth_targets_local[oracle_truth_mask_local],
            )
        if float(oracle_mode_best_weight) > 0.0 and bool(oracle_mode_best_mask_local.any().item()):
            loss_local = loss_local + float(oracle_mode_best_weight) * F.binary_cross_entropy_with_logits(
                outputs["oracle_preview_mode_best_logit"][oracle_mode_best_mask_local],
                oracle_mode_best_targets_local[oracle_mode_best_mask_local],
            )
        if float(oracle_rank_weight) > 0.0 and bool(oracle_rank_mask_local.any().item()):
            loss_local = loss_local + float(oracle_rank_weight) * F.smooth_l1_loss(
                torch.sigmoid(outputs["oracle_preview_rank_logit"][oracle_rank_mask_local]),
                oracle_rank_targets_local[oracle_rank_mask_local],
            )
        if float(oracle_stability_weight) > 0.0 and bool(oracle_stability_mask_local.any().item()):
            loss_local = loss_local + float(oracle_stability_weight) * F.binary_cross_entropy_with_logits(
                outputs["oracle_preview_stability_logit"][oracle_stability_mask_local],
                oracle_stability_targets_local[oracle_stability_mask_local],
            )
        if float(oracle_coverage_weight) > 0.0 and bool(oracle_coverage_mask_local.any().item()):
            loss_local = loss_local + float(oracle_coverage_weight) * F.binary_cross_entropy_with_logits(
                outputs["oracle_state_coverage_logit"][oracle_coverage_mask_local],
                oracle_coverage_targets_local[oracle_coverage_mask_local],
            )
        return loss_local

    for _epoch in range(max(1, int(epochs))):
        model.train()
        outputs = model(
            x_train_n,
            path_x_train_n,
            path_valid_mask_train,
            preview_x_train_n,
            preview_path_indices_train,
            preview_mask_train,
            provenance_x_train_n,
            provenance_mask_train,
        )
        loss = _loss_fn(
            outputs,
            y_preview_utility_train,
            repair_preview_mask_train,
            build_preview_mask_train,
            repair_pairwise_pairs_train,
            build_pairwise_pairs_train,
            y_repair_path_train,
            repair_path_mask_train,
            y_oracle_path_train,
            oracle_path_mask_train,
            y_oracle_relation_train,
            oracle_relation_mask_train,
            y_oracle_mode_train,
            oracle_mode_mask_train,
            path_valid_mask_train,
            preview_mask_train,
            y_repair_state_value_train_n,
            repair_state_value_mask_train,
            y_build_state_value_train_n,
            build_state_value_mask_train,
            y_preview_value_train,
            y_preview_regret_train,
            y_preview_q_train,
            preview_value_mask_train,
            common_pairwise_pairs_train,
            y_common_state_value_train_n,
            y_oracle_truth_train,
            oracle_truth_mask_train,
            y_oracle_mode_best_train,
            oracle_mode_best_mask_train,
            y_oracle_rank_train,
            oracle_rank_mask_train,
            y_oracle_stability_train,
            oracle_stability_mask_train,
            y_oracle_coverage_train,
            oracle_coverage_mask_train,
        )
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        opt.step()

        model.eval()
        with torch.no_grad():
            outputs_val = model(
                x_val_n,
                path_x_val_n,
                path_valid_mask_val,
                preview_x_val_n,
                preview_path_indices_val,
                preview_mask_val,
                provenance_x_val_n,
                provenance_mask_val,
            )
            val_loss = _loss_fn(
                outputs_val,
                y_preview_utility_val,
                repair_preview_mask_val,
                build_preview_mask_val,
                repair_pairwise_pairs_val,
                build_pairwise_pairs_val,
                y_repair_path_val,
                repair_path_mask_val,
                y_oracle_path_val,
                oracle_path_mask_val,
                y_oracle_relation_val,
                oracle_relation_mask_val,
                y_oracle_mode_val,
                oracle_mode_mask_val,
                path_valid_mask_val,
                preview_mask_val,
                y_repair_state_value_val_n,
                repair_state_value_mask_val,
                y_build_state_value_val_n,
                build_state_value_mask_val,
                y_preview_value_val,
                y_preview_regret_val,
                y_preview_q_val,
                preview_value_mask_val,
                common_pairwise_pairs_val,
                y_common_state_value_val_n,
                y_oracle_truth_val,
                oracle_truth_mask_val,
                y_oracle_mode_best_val,
                oracle_mode_best_mask_val,
                y_oracle_rank_val,
                oracle_rank_mask_val,
                y_oracle_stability_val,
                oracle_stability_mask_val,
                y_oracle_coverage_val,
                oracle_coverage_mask_val,
            )
            val_loss_f = float(val_loss.item())
        if val_loss_f + 1.0e-8 < best_val:
            best_val = val_loss_f
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        train_out = model(
            x_train_n,
            path_x_train_n,
            path_valid_mask_train,
            preview_x_train_n,
            preview_path_indices_train,
            preview_mask_train,
            provenance_x_train_n,
            provenance_mask_train,
        )
        val_out = model(
            x_val_n,
            path_x_val_n,
            path_valid_mask_val,
            preview_x_val_n,
            preview_path_indices_val,
            preview_mask_val,
            provenance_x_val_n,
            provenance_mask_val,
        )
        train_metrics: dict[str, float] = {}
        val_metrics: dict[str, float] = {}
        train_metrics.update(_slate_rank_metrics(
            train_out["repair_preview_score"],
            y_preview_utility_train,
            repair_preview_mask_train,
            repair_pairwise_pairs_train,
            prefix="repair_candidate",
        ))
        val_metrics.update(_slate_rank_metrics(
            val_out["repair_preview_score"],
            y_preview_utility_val,
            repair_preview_mask_val,
            repair_pairwise_pairs_val,
            prefix="repair_candidate",
        ))
        train_metrics.update(_slate_rank_metrics(
            train_out["build_preview_score"],
            y_preview_utility_train,
            build_preview_mask_train,
            build_pairwise_pairs_train,
            prefix="build_candidate",
        ))
        train_metrics.update(_slate_rank_metrics(
            train_out["common_preview_q"],
            y_preview_q_train,
            preview_mask_train,
            common_pairwise_pairs_train,
            prefix="common_candidate",
        ))
        val_metrics.update(_slate_rank_metrics(
            val_out["build_preview_score"],
            y_preview_utility_val,
            build_preview_mask_val,
            build_pairwise_pairs_val,
            prefix="build_candidate",
        ))
        val_metrics.update(_slate_rank_metrics(
            val_out["common_preview_q"],
            y_preview_q_val,
            preview_mask_val,
            common_pairwise_pairs_val,
            prefix="common_candidate",
        ))
        if bool(repair_path_mask_train.any().item()):
            train_metrics.update(_path_metrics_from_logits(
                train_out["path_logits"][repair_path_mask_train],
                y_repair_path_train[repair_path_mask_train],
                path_valid_mask_train[repair_path_mask_train],
            ))
        if bool(oracle_path_mask_train.any().item()):
            oracle_path_metrics = _path_metrics_from_logits(
                train_out["path_logits"][oracle_path_mask_train],
                y_oracle_path_train[oracle_path_mask_train],
                path_valid_mask_train[oracle_path_mask_train],
            )
            train_metrics.update({f"oracle_{k}": v for k, v in oracle_path_metrics.items()})
        if bool(repair_path_mask_val.any().item()):
            val_metrics.update(_path_metrics_from_logits(
                val_out["path_logits"][repair_path_mask_val],
                y_repair_path_val[repair_path_mask_val],
                path_valid_mask_val[repair_path_mask_val],
            ))
        if bool(oracle_path_mask_val.any().item()):
            oracle_path_metrics = _path_metrics_from_logits(
                val_out["path_logits"][oracle_path_mask_val],
                y_oracle_path_val[oracle_path_mask_val],
                path_valid_mask_val[oracle_path_mask_val],
            )
            val_metrics.update({f"oracle_{k}": v for k, v in oracle_path_metrics.items()})
        if bool(oracle_relation_mask_train.any().item()):
            train_metrics.update(_flat_classification_metrics(
                train_out["path_relation_logits"][oracle_relation_mask_train],
                y_oracle_relation_train[oracle_relation_mask_train],
                prefix="oracle_path_relation",
            ))
        if bool(oracle_relation_mask_val.any().item()):
            val_metrics.update(_flat_classification_metrics(
                val_out["path_relation_logits"][oracle_relation_mask_val],
                y_oracle_relation_val[oracle_relation_mask_val],
                prefix="oracle_path_relation",
            ))
        if bool(oracle_mode_mask_train.any().item()):
            train_metrics.update(_flat_classification_metrics(
                train_out["path_mode_logits"][oracle_mode_mask_train],
                y_oracle_mode_train[oracle_mode_mask_train],
                prefix="oracle_path_mode",
            ))
        if bool(oracle_mode_mask_val.any().item()):
            val_metrics.update(_flat_classification_metrics(
                val_out["path_mode_logits"][oracle_mode_mask_val],
                y_oracle_mode_val[oracle_mode_mask_val],
                prefix="oracle_path_mode",
            ))
        if bool(repair_state_value_mask_train.any().item()):
            train_metrics.update(_regression_metrics(
                train_out["repair_state_value"][repair_state_value_mask_train] * repair_state_std + repair_state_mean,
                y_repair_state_value_train[repair_state_value_mask_train],
                prefix="repair_state_value",
            ))
        if bool(repair_state_value_mask_val.any().item()):
            val_metrics.update(_regression_metrics(
                val_out["repair_state_value"][repair_state_value_mask_val] * repair_state_std + repair_state_mean,
                y_repair_state_value_val[repair_state_value_mask_val],
                prefix="repair_state_value",
            ))
        if bool(build_state_value_mask_train.any().item()):
            train_metrics.update(_regression_metrics(
                train_out["build_state_value"][build_state_value_mask_train] * build_state_std + build_state_mean,
                y_build_state_value_train[build_state_value_mask_train],
                prefix="build_state_value",
            ))
        if bool(build_state_value_mask_val.any().item()):
            val_metrics.update(_regression_metrics(
                val_out["build_state_value"][build_state_value_mask_val] * build_state_std + build_state_mean,
                y_build_state_value_val[build_state_value_mask_val],
                prefix="build_state_value",
            ))
        if bool(preview_value_mask_train.any().item()):
            train_metrics.update(_regression_metrics(
                train_out["preview_value"][preview_value_mask_train],
                y_preview_value_train[preview_value_mask_train],
                prefix="preview_value",
            ))
        if bool(preview_value_mask_val.any().item()):
            val_metrics.update(_regression_metrics(
                val_out["preview_value"][preview_value_mask_val],
                y_preview_value_val[preview_value_mask_val],
                prefix="preview_value",
            ))
        if bool(preview_mask_train.any().item()):
            train_metrics.update(_regression_metrics(
                train_out["preview_regret"][preview_mask_train],
                y_preview_regret_train[preview_mask_train],
                prefix="preview_regret",
            ))
            train_metrics.update(_regression_metrics(
                train_out["common_preview_q"][preview_mask_train],
                y_preview_q_train[preview_mask_train],
                prefix="common_preview_q",
            ))
        if bool(oracle_truth_mask_train.any().item()):
            train_metrics.update(_binary_classification_metrics(
                train_out["oracle_preview_truth_logit"][oracle_truth_mask_train],
                y_oracle_truth_train[oracle_truth_mask_train].to(dtype=torch.long),
                prefix="oracle_preview_truth",
            ))
        if bool(oracle_mode_best_mask_train.any().item()):
            train_metrics.update(_binary_classification_metrics(
                train_out["oracle_preview_mode_best_logit"][oracle_mode_best_mask_train],
                y_oracle_mode_best_train[oracle_mode_best_mask_train].to(dtype=torch.long),
                prefix="oracle_preview_mode_best",
            ))
        if bool(oracle_rank_mask_train.any().item()):
            train_metrics.update(_regression_metrics(
                torch.sigmoid(train_out["oracle_preview_rank_logit"][oracle_rank_mask_train]),
                y_oracle_rank_train[oracle_rank_mask_train],
                prefix="oracle_preview_rank",
            ))
        if bool(oracle_stability_mask_train.any().item()):
            train_metrics.update(_binary_classification_metrics(
                train_out["oracle_preview_stability_logit"][oracle_stability_mask_train],
                y_oracle_stability_train[oracle_stability_mask_train].to(dtype=torch.long),
                prefix="oracle_preview_stability",
            ))
        if bool(oracle_coverage_mask_train.any().item()):
            train_metrics.update(_binary_classification_metrics(
                train_out["oracle_state_coverage_logit"][oracle_coverage_mask_train],
                y_oracle_coverage_train[oracle_coverage_mask_train].to(dtype=torch.long),
                prefix="oracle_state_coverage",
            ))
        if bool(preview_mask_val.any().item()):
            val_metrics.update(_regression_metrics(
                val_out["preview_regret"][preview_mask_val],
                y_preview_regret_val[preview_mask_val],
                prefix="preview_regret",
            ))
            val_metrics.update(_regression_metrics(
                val_out["common_preview_q"][preview_mask_val],
                y_preview_q_val[preview_mask_val],
                prefix="common_preview_q",
            ))
        if bool(oracle_truth_mask_val.any().item()):
            val_metrics.update(_binary_classification_metrics(
                val_out["oracle_preview_truth_logit"][oracle_truth_mask_val],
                y_oracle_truth_val[oracle_truth_mask_val].to(dtype=torch.long),
                prefix="oracle_preview_truth",
            ))
        if bool(oracle_mode_best_mask_val.any().item()):
            val_metrics.update(_binary_classification_metrics(
                val_out["oracle_preview_mode_best_logit"][oracle_mode_best_mask_val],
                y_oracle_mode_best_val[oracle_mode_best_mask_val].to(dtype=torch.long),
                prefix="oracle_preview_mode_best",
            ))
        if bool(oracle_rank_mask_val.any().item()):
            val_metrics.update(_regression_metrics(
                torch.sigmoid(val_out["oracle_preview_rank_logit"][oracle_rank_mask_val]),
                y_oracle_rank_val[oracle_rank_mask_val],
                prefix="oracle_preview_rank",
            ))
        if bool(oracle_stability_mask_val.any().item()):
            val_metrics.update(_binary_classification_metrics(
                val_out["oracle_preview_stability_logit"][oracle_stability_mask_val],
                y_oracle_stability_val[oracle_stability_mask_val].to(dtype=torch.long),
                prefix="oracle_preview_stability",
            ))
        if bool(oracle_coverage_mask_val.any().item()):
            val_metrics.update(_binary_classification_metrics(
                val_out["oracle_state_coverage_logit"][oracle_coverage_mask_val],
                y_oracle_coverage_val[oracle_coverage_mask_val].to(dtype=torch.long),
                prefix="oracle_state_coverage",
            ))
        train_metrics.update(_regression_metrics(
            train_out["value_pred"] * common_state_std + common_state_mean,
            y_common_state_value_train,
            prefix="common_state_value",
        ))
        val_metrics.update(_regression_metrics(
            val_out["value_pred"] * common_state_std + common_state_mean,
            y_common_state_value_val,
            prefix="common_state_value",
        ))

    return {
        "model_kind": REPAIR_CRITIC_SHARED_CANDIDATE_MODEL_KIND,
        "feature_names": feature_names,
        "path_feature_names": path_feature_names,
        "preview_feature_names": preview_feature_names,
        "provenance_feature_names": provenance_feature_names,
        "hidden_dim": int(hidden_dim),
        "feature_mean": mean.detach().cpu(),
        "feature_std": std.detach().cpu(),
        "path_feature_mean": torch.as_tensor(path_mean.detach().cpu(), dtype=torch.float32),
        "path_feature_std": torch.as_tensor(path_std.detach().cpu(), dtype=torch.float32),
        "preview_feature_mean": torch.as_tensor(preview_mean.detach().cpu(), dtype=torch.float32),
        "preview_feature_std": torch.as_tensor(preview_std.detach().cpu(), dtype=torch.float32),
        "provenance_feature_mean": torch.as_tensor(provenance_mean.detach().cpu(), dtype=torch.float32),
        "provenance_feature_std": torch.as_tensor(provenance_std.detach().cpu(), dtype=torch.float32),
        "shared_candidate_dual_trained": True,
        "shared_candidate_dual_target": "within_route_best_exact_candidate",
        "shared_candidate_repair_rank_weight": float(max(0.0, repair_rank_weight)),
        "shared_candidate_build_rank_weight": float(max(0.0, build_rank_weight)),
        "shared_candidate_common_rank_weight": float(max(0.0, common_rank_weight)),
        "shared_candidate_common_value_target": f"immediate_refine_{float(common_value_weight):.3f}x_continuation_minus_{float(common_regret_weight):.3f}x_regret",
        "shared_candidate_common_q_weight": float(max(0.0, common_q_weight)),
        "shared_candidate_preview_value_weight": float(max(0.0, preview_value_weight)),
        "shared_candidate_preview_regret_weight": float(max(0.0, preview_regret_weight)),
        "shared_candidate_common_state_value_weight": float(max(0.0, common_state_value_weight)),
        "shared_candidate_oracle_path_weight": float(max(0.0, oracle_path_weight)),
        "shared_candidate_oracle_relation_weight": float(max(0.0, oracle_relation_weight)),
        "shared_candidate_oracle_mode_weight": float(max(0.0, oracle_mode_weight)),
        "shared_candidate_oracle_truth_weight": float(max(0.0, oracle_truth_weight)),
        "shared_candidate_oracle_mode_best_weight": float(max(0.0, oracle_mode_best_weight)),
        "shared_candidate_oracle_rank_weight": float(max(0.0, oracle_rank_weight)),
        "shared_candidate_oracle_stability_weight": float(max(0.0, oracle_stability_weight)),
        "shared_candidate_oracle_coverage_weight": float(max(0.0, oracle_coverage_weight)),
        "repair_action_names": sorted(name for name in repair_action_names_seen if name),
        "build_action_names": sorted(name for name in build_action_names_seen if name),
        "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "metrics": {
            "n_examples": int(len(training_rows)),
            "n_repair_route_examples": int((repair_preview_mask.sum(dim=-1) >= 2).sum().item()),
            "n_build_route_examples": int((build_preview_mask.sum(dim=-1) >= 2).sum().item()),
            "n_preview_examples": int(preview_mask.sum().item()),
            "n_provenance_tuples": int(provenance_mask.sum().item()),
            "n_oracle_path_examples": int(oracle_path_mask.sum().item()),
            "n_oracle_relation_examples": int(oracle_relation_mask.sum().item()),
            "n_oracle_mode_examples": int(oracle_mode_mask.sum().item()),
            "n_oracle_truth_examples": int(oracle_truth_mask.sum().item()),
            "n_oracle_mode_best_examples": int(oracle_mode_best_mask.sum().item()),
            "n_oracle_rank_examples": int(oracle_rank_mask.sum().item()),
            "n_oracle_stability_examples": int(oracle_stability_mask.sum().item()),
            "n_oracle_coverage_examples": int(oracle_coverage_mask.sum().item()),
            "n_train": int(x_train.shape[0]),
            "n_val": int(x_val.shape[0]),
            "best_val_loss": float(best_val),
            "reward_source_counts": dict(reward_source_counts),
            "train": train_metrics,
            "val": val_metrics,
        },
    }


def train_repair_build_route_comparator(
    rows: Sequence[Any],
    *,
    hidden_dim: int = 32,
    epochs: int = 120,
    lr: float = 5.0e-3,
    weight_decay: float = 1.0e-4,
    val_fraction: float = 0.2,
    seed: int = 0,
    margin_floor: float = 1.0e-3,
    margin_loss_weight: float = 0.25,
    repair_tuple_bundle: dict[str, Any] | None = None,
    build_tuple_bundle: dict[str, Any] | None = None,
    init_bundle: dict[str, Any] | None = None,
) -> dict[str, Any]:
    torch.manual_seed(int(seed))
    training_rows = _build_repair_build_route_rows(
        rows,
        margin_floor=float(margin_floor),
        repair_tuple_bundle=repair_tuple_bundle,
        build_tuple_bundle=build_tuple_bundle,
    )
    if len(training_rows) < 8:
        raise ValueError("Need at least 8 repair/build route rows to train a comparator.")

    feature_names = list(tuple(REPAIR_CRITIC_FEATURE_NAMES) + tuple(REPAIR_ROUTE_COMPARE_EXTRA_FEATURE_NAMES))
    n_rows = len(training_rows)
    x = torch.tensor(
        [[float(row["features"].get(name, 0.0)) for name in feature_names] for row in training_rows],
        dtype=torch.float32,
    )
    y_route = torch.tensor([int(row.get("route_target", 0)) for row in training_rows], dtype=torch.float32)
    y_margin = torch.tensor([float(row.get("route_margin", 0.0) or 0.0) for row in training_rows], dtype=torch.float32)
    sample_weight = torch.tensor([float(row.get("sample_weight", 1.0) or 1.0) for row in training_rows], dtype=torch.float32)

    train_idx, val_idx = _split_indices(n_rows, float(val_fraction), int(seed))
    x_train = x[train_idx]
    x_val = x[val_idx] if len(val_idx) > 0 else x_train
    y_route_train = y_route[train_idx]
    y_route_val = y_route[val_idx] if len(val_idx) > 0 else y_route_train
    y_margin_train = y_margin[train_idx]
    y_margin_val = y_margin[val_idx] if len(val_idx) > 0 else y_margin_train
    weight_train = sample_weight[train_idx]
    weight_val = sample_weight[val_idx] if len(val_idx) > 0 else weight_train

    mean, std = _compute_feature_stats(x_train)
    x_train_n = _normalize_inputs(x_train, mean, std)
    x_val_n = _normalize_inputs(x_val, mean, std)

    model = _RepairRouteCompareNet(int(x.shape[1]), int(hidden_dim)).to(dtype=torch.float32)
    if isinstance(init_bundle, dict) and str(init_bundle.get("model_kind", "")) == REPAIR_CRITIC_ROUTE_COMPARE_MODEL_KIND:
        _maybe_init_model_from_bundle(model, init_bundle)
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))

    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_val = float("inf")
    patience = max(20, int(0.20 * max(1, int(epochs))))
    bad_epochs = 0

    def _route_loss(
        outputs: dict[str, torch.Tensor],
        target_route: torch.Tensor,
        target_margin: torch.Tensor,
        target_weight: torch.Tensor,
    ) -> torch.Tensor:
        weight = target_weight / target_weight.mean().clamp_min(1.0e-6)
        route_loss = F.binary_cross_entropy_with_logits(
            outputs["route_logit"],
            target_route,
            reduction="none",
        )
        margin_loss = F.smooth_l1_loss(
            outputs["margin_pred"],
            target_margin,
            reduction="none",
        )
        return torch.mean(weight * (route_loss + float(margin_loss_weight) * margin_loss))

    for _epoch in range(max(1, int(epochs))):
        model.train()
        outputs = model(x_train_n)
        loss = _route_loss(outputs, y_route_train, y_margin_train, weight_train)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        opt.step()

        model.eval()
        with torch.no_grad():
            outputs_val = model(x_val_n)
            val_loss = _route_loss(outputs_val, y_route_val, y_margin_val, weight_val)
            val_loss_f = float(val_loss.item())
        if val_loss_f + 1.0e-8 < best_val:
            best_val = val_loss_f
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        train_out = model(x_train_n)
        val_out = model(x_val_n)
        train_metrics = {}
        train_metrics.update(_binary_classification_metrics(train_out["route_logit"], y_route_train.to(dtype=torch.long), prefix="route"))
        train_metrics.update(_regression_metrics(train_out["margin_pred"], y_margin_train, prefix="route_margin"))
        val_metrics = {}
        val_metrics.update(_binary_classification_metrics(val_out["route_logit"], y_route_val.to(dtype=torch.long), prefix="route"))
        val_metrics.update(_regression_metrics(val_out["margin_pred"], y_margin_val, prefix="route_margin"))

    return {
        "model_kind": REPAIR_CRITIC_ROUTE_COMPARE_MODEL_KIND,
        "feature_names": feature_names,
        "head_names": [],
        "hidden_dim": int(hidden_dim),
        "feature_mean": mean.detach().cpu(),
        "feature_std": std.detach().cpu(),
        "repair_build_route_compare_trained": True,
        "repair_build_route_compare_target": "same_parent_best_exact_margin",
        "repair_build_route_compare_margin_floor": float(max(0.0, margin_floor)),
        "repair_build_route_compare_margin_loss_weight": float(max(0.0, margin_loss_weight)),
        "repair_build_route_compare_uses_repair_tuple_features": bool(isinstance(repair_tuple_bundle, dict)),
        "repair_build_route_compare_uses_build_tuple_features": bool(isinstance(build_tuple_bundle, dict)),
        "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "metrics": {
            "n_examples": int(len(training_rows)),
            "n_train": int(x_train.shape[0]),
            "n_val": int(x_val.shape[0]),
            "n_repair_targets": int((y_route > 0.5).sum().item()),
            "n_build_targets": int((y_route <= 0.5).sum().item()),
            "best_val_loss": float(best_val),
            "train": train_metrics,
            "val": val_metrics,
        },
    }
