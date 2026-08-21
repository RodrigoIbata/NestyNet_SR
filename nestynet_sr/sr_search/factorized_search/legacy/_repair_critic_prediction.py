# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

from __future__ import annotations

"""Serialization and inference entry points for repair-critic models."""

import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch

from nestynet_sr.sr_search.factorized_search.engine.signals import PathStateFeatures
from nestynet_sr.sr_search.factorized_search.policy.features import (
    coerce_repair_feature_record,
    coerce_repair_feature_row,
)
from nestynet_sr.sr_search.factorized_search.shared_candidate import shared_candidate_row_dict

from ._repair_critic_features import (
    BUILD_TUPLE_PREVIEW_FEATURE_NAMES,
    REPAIR_CRITIC_ACTION_ROUTE_MAP,
    REPAIR_CRITIC_BUILD_TUPLE_MODEL_KIND,
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
    REPAIR_CRITIC_UNIFIED_CANDIDATE_MODEL_KIND,
    REPAIR_ROUTE_COMPARE_EXTRA_FEATURE_NAMES,
    UNIFIED_CANDIDATE_PREVIEW_FEATURE_NAMES,
    _REPAIR_CRITIC_DEFAULT_REPAIR_ACTION_SET,
    _build_build_tuple_rows,
    _build_repair_build_route_rows,
    _group_build_preview_rows,
    _group_repair_preview_rows,
    _match_candidate_path_index,
    _match_path_mode_index,
    _normalize_action_name,
    _to_float,
    build_preview_feature_vector,
    repair_critic_feature_vector,
    repair_path_feature_vector,
    repair_preview_feature_vector,
    unified_candidate_preview_feature_vector,
)

from ._repair_critic_models import (
    _BuildTupleRankerNet,
    _RepairControllerSharedNet,
    _RepairCriticNet,
    _RepairRouteCompareNet,
    _SharedCandidateDualRankerNet,
    _aux_predictions_from_logits,
    _bundle_kind_from_payload,
    _masked_path_probs,
    _maybe_init_model_from_bundle,
    _normalize_inputs,
)

def save_repair_critic_bundle(bundle: dict[str, Any], path: str | Path) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, out_path)


def load_repair_critic_bundle(path: str | Path) -> dict[str, Any]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    payload = dict(payload)
    feature_names = list(payload.get("feature_names", list(REPAIR_CRITIC_FEATURE_NAMES)))
    head_names = list(payload.get("head_names", list(REPAIR_CRITIC_HEAD_NAMES)))
    hidden_dim = int(payload.get("hidden_dim", 32))
    model_kind = _bundle_kind_from_payload(payload)
    payload["model_kind"] = model_kind
    payload["feature_names"] = feature_names
    payload["head_names"] = head_names
    payload["feature_mean"] = torch.as_tensor(
        payload.get("feature_mean", torch.zeros(len(feature_names))),
        dtype=torch.float32,
    )
    payload["feature_std"] = torch.as_tensor(
        payload.get("feature_std", torch.ones(len(feature_names))),
        dtype=torch.float32,
    )
    if model_kind == REPAIR_CRITIC_BUILD_TUPLE_MODEL_KIND:
        build_preview_feature_names = list(payload.get("build_preview_feature_names", list(BUILD_TUPLE_PREVIEW_FEATURE_NAMES)))
        model = _BuildTupleRankerNet(len(feature_names), len(build_preview_feature_names), hidden_dim).to(dtype=torch.float32)
        model.load_state_dict(payload.get("model_state_dict", {}), strict=True)
        model.eval()
        payload["build_preview_feature_names"] = build_preview_feature_names
        payload["build_preview_feature_mean"] = torch.as_tensor(
            payload.get("build_preview_feature_mean", torch.zeros(len(build_preview_feature_names))),
            dtype=torch.float32,
        )
        payload["build_preview_feature_std"] = torch.as_tensor(
            payload.get("build_preview_feature_std", torch.ones(len(build_preview_feature_names))),
            dtype=torch.float32,
        )
        payload["build_tuple_ranker_trained"] = bool(payload.get("build_tuple_ranker_trained", False))
        payload["build_tuple_ranker_target"] = str(payload.get("build_tuple_ranker_target", "") or "")
        payload["build_tuple_ranker_score_gap_floor"] = float(max(0.0, _to_float(payload.get("build_tuple_ranker_score_gap_floor", 0.0), 0.0)))
        payload["build_tuple_ranker_state_value_weight"] = float(max(0.0, _to_float(payload.get("build_tuple_ranker_state_value_weight", 0.0), 0.0)))
        payload["model"] = model
        return payload
    if model_kind == REPAIR_CRITIC_ROUTE_COMPARE_MODEL_KIND:
        model = _RepairRouteCompareNet(len(feature_names), hidden_dim).to(dtype=torch.float32)
        model.load_state_dict(payload.get("model_state_dict", {}), strict=True)
        model.eval()
        payload["repair_build_route_compare_trained"] = bool(payload.get("repair_build_route_compare_trained", False))
        payload["repair_build_route_compare_target"] = str(payload.get("repair_build_route_compare_target", "") or "")
        payload["repair_build_route_compare_margin_floor"] = float(max(0.0, _to_float(payload.get("repair_build_route_compare_margin_floor", 0.0), 0.0)))
        payload["repair_build_route_compare_margin_loss_weight"] = float(max(0.0, _to_float(payload.get("repair_build_route_compare_margin_loss_weight", 0.25), 0.25)))
        payload["model"] = model
        return payload
    if model_kind == REPAIR_CRITIC_SHARED_CANDIDATE_MODEL_KIND:
        path_feature_names = list(payload.get("path_feature_names", list(REPAIR_CRITIC_PATH_FEATURE_NAMES)))
        preview_feature_names = list(payload.get("preview_feature_names", list(UNIFIED_CANDIDATE_PREVIEW_FEATURE_NAMES)))
        provenance_feature_names = list(payload.get("provenance_feature_names", list(preview_feature_names)))
        model = _SharedCandidateDualRankerNet(
            len(feature_names),
            hidden_dim,
            path_input_dim=len(path_feature_names),
            preview_input_dim=len(preview_feature_names),
            provenance_input_dim=len(provenance_feature_names),
        ).to(dtype=torch.float32)
        _maybe_init_model_from_bundle(model, payload)
        model.eval()
        payload["path_feature_names"] = path_feature_names
        payload["preview_feature_names"] = preview_feature_names
        payload["provenance_feature_names"] = provenance_feature_names
        payload["path_feature_mean"] = torch.as_tensor(
            payload.get("path_feature_mean", torch.zeros(len(path_feature_names))),
            dtype=torch.float32,
        )
        payload["path_feature_std"] = torch.as_tensor(
            payload.get("path_feature_std", torch.ones(len(path_feature_names))),
            dtype=torch.float32,
        )
        payload["preview_feature_mean"] = torch.as_tensor(
            payload.get("preview_feature_mean", torch.zeros(len(preview_feature_names))),
            dtype=torch.float32,
        )
        payload["preview_feature_std"] = torch.as_tensor(
            payload.get("preview_feature_std", torch.ones(len(preview_feature_names))),
            dtype=torch.float32,
        )
        payload["provenance_feature_mean"] = torch.as_tensor(
            payload.get("provenance_feature_mean", torch.zeros(len(provenance_feature_names))),
            dtype=torch.float32,
        )
        payload["provenance_feature_std"] = torch.as_tensor(
            payload.get("provenance_feature_std", torch.ones(len(provenance_feature_names))),
            dtype=torch.float32,
        )
        payload["shared_candidate_dual_trained"] = bool(payload.get("shared_candidate_dual_trained", False))
        payload["shared_candidate_dual_target"] = str(payload.get("shared_candidate_dual_target", "") or "")
        payload["repair_action_names"] = list(payload.get("repair_action_names", []))
        payload["build_action_names"] = list(payload.get("build_action_names", []))
        payload["model_kind"] = str(model_kind)
        payload["model"] = model
        return payload
    if model_kind in {REPAIR_CRITIC_SHARED_MODEL_KIND, REPAIR_CRITIC_UNIFIED_CANDIDATE_MODEL_KIND}:
        path_feature_names = list(payload.get("path_feature_names", list(REPAIR_CRITIC_PATH_FEATURE_NAMES)))
        preview_feature_names = list(payload.get("preview_feature_names", list(REPAIR_CRITIC_PREVIEW_FEATURE_NAMES) if payload.get("repair_tuple_ranker_trained", False) else []))
        provenance_feature_names = list(payload.get("provenance_feature_names", list(preview_feature_names)))
        macro_action_names = list(payload.get("macro_action_names", list(REPAIR_CRITIC_MACRO_ACTION_NAMES)))
        route_names = list(payload.get("route_names", list(REPAIR_CRITIC_ROUTE_NAMES)))
        model = _RepairControllerSharedNet(
            len(feature_names),
            hidden_dim,
            n_macro_actions=len(macro_action_names),
            n_routes=len(route_names),
            path_input_dim=len(path_feature_names),
            preview_input_dim=len(preview_feature_names),
            provenance_input_dim=len(provenance_feature_names),
        ).to(dtype=torch.float32)
        _maybe_init_model_from_bundle(model, payload)
        model.eval()
        payload["path_feature_names"] = path_feature_names
        payload["preview_feature_names"] = preview_feature_names
        payload["provenance_feature_names"] = provenance_feature_names
        payload["macro_action_names"] = macro_action_names
        payload["route_names"] = route_names
        payload["mode_names"] = list(payload.get("mode_names", list(REPAIR_CRITIC_MODE_NAMES)))
        payload["path_relation_names"] = list(payload.get("path_relation_names", list(REPAIR_CRITIC_PATH_RELATION_NAMES)))
        payload["path_feature_mean"] = torch.as_tensor(
            payload.get("path_feature_mean", torch.zeros(len(path_feature_names))),
            dtype=torch.float32,
        )
        payload["path_feature_std"] = torch.as_tensor(
            payload.get("path_feature_std", torch.ones(len(path_feature_names))),
            dtype=torch.float32,
        )
        payload["preview_feature_mean"] = torch.as_tensor(
            payload.get("preview_feature_mean", torch.zeros(len(preview_feature_names))),
            dtype=torch.float32,
        )
        payload["preview_feature_std"] = torch.as_tensor(
            payload.get("preview_feature_std", torch.ones(len(preview_feature_names))),
            dtype=torch.float32,
        )
        payload["provenance_feature_mean"] = torch.as_tensor(
            payload.get("provenance_feature_mean", torch.zeros(len(provenance_feature_names))),
            dtype=torch.float32,
        )
        payload["provenance_feature_std"] = torch.as_tensor(
            payload.get("provenance_feature_std", torch.ones(len(provenance_feature_names))),
            dtype=torch.float32,
        )
        payload["actor_critic_reward_target"] = str(payload.get("actor_critic_reward_target", "immediate") or "immediate")
        payload["actor_critic_reward_mean"] = float(payload.get("actor_critic_reward_mean", 0.0))
        payload["actor_critic_reward_std"] = float(max(1.0e-6, _to_float(payload.get("actor_critic_reward_std", 1.0), 1.0)))
        payload["repair_slate_ranker_trained"] = bool(payload.get("repair_slate_ranker_trained", False))
        payload["repair_slate_ranker_target"] = str(payload.get("repair_slate_ranker_target", "") or "")
        payload["repair_slate_action_names"] = list(payload.get("repair_slate_action_names", []))
        payload["repair_tuple_ranker_trained"] = bool(payload.get("repair_tuple_ranker_trained", False))
        payload["repair_tuple_ranker_target"] = str(payload.get("repair_tuple_ranker_target", "") or "")
        payload["repair_tuple_action_names"] = list(payload.get("repair_tuple_action_names", []))
        payload["unified_candidate_ranker_trained"] = bool(payload.get("unified_candidate_ranker_trained", False))
        payload["unified_candidate_ranker_target"] = str(payload.get("unified_candidate_ranker_target", "") or "")
        payload["unified_candidate_route_tau"] = float(max(1.0e-6, _to_float(payload.get("unified_candidate_route_tau", 1.0), 1.0)))
        payload["unified_candidate_action_names"] = list(payload.get("unified_candidate_action_names", []))
        payload["repair_tuple_preview_value_target"] = str(payload.get("repair_tuple_preview_value_target", "") or "")
        payload["repair_tuple_child_value_lambda"] = float(max(0.0, _to_float(payload.get("repair_tuple_child_value_lambda", 0.0), 0.0)))
        payload["repair_tuple_regret_target"] = str(payload.get("repair_tuple_regret_target", "") or "")
        payload["repair_tuple_regret_weight"] = float(max(0.0, _to_float(payload.get("repair_tuple_regret_weight", 1.0), 1.0)))
        payload["macro_head_trained"] = bool(payload.get("macro_head_trained", False))
        payload["route_head_trained"] = bool(payload.get("route_head_trained", False))
        payload["q_head_trained"] = bool(payload.get("q_head_trained", False))
        payload["value_head_trained"] = bool(payload.get("value_head_trained", False))
        payload["regret_head_trained"] = bool(payload.get("regret_head_trained", payload.get("repair_tuple_ranker_trained", False)))
        payload["path_head_trained"] = bool(payload.get("path_head_trained", False))
        payload["path_action_head_trained"] = bool(payload.get("path_action_head_trained", False))
        payload["path_relation_head_trained"] = bool(payload.get("path_relation_head_trained", False))
        payload["path_mode_head_trained"] = bool(payload.get("path_mode_head_trained", False))
        payload["path_improve_head_trained"] = bool(payload.get("path_improve_head_trained", False))
        payload["model_kind"] = str(model_kind)
        payload["model"] = model
        return payload
    model = _RepairCriticNet(len(feature_names), hidden_dim).to(dtype=torch.float32)
    model.load_state_dict(payload.get("model_state_dict", {}), strict=True)
    model.eval()
    payload["path_feature_names"] = list(payload.get("path_feature_names", list(REPAIR_CRITIC_PATH_FEATURE_NAMES)))
    payload["preview_feature_names"] = list(payload.get("preview_feature_names", []))
    payload["provenance_feature_names"] = list(payload.get("provenance_feature_names", list(payload["preview_feature_names"])))
    payload["macro_action_names"] = list(payload.get("macro_action_names", list(REPAIR_CRITIC_MACRO_ACTION_NAMES)))
    payload["route_names"] = list(payload.get("route_names", list(REPAIR_CRITIC_ROUTE_NAMES)))
    payload["path_feature_mean"] = torch.as_tensor(
        payload.get("path_feature_mean", torch.zeros(len(payload["path_feature_names"]))),
        dtype=torch.float32,
    )
    payload["path_feature_std"] = torch.as_tensor(
        payload.get("path_feature_std", torch.ones(len(payload["path_feature_names"]))),
        dtype=torch.float32,
    )
    payload["preview_feature_mean"] = torch.as_tensor(
        payload.get("preview_feature_mean", torch.zeros(len(payload["preview_feature_names"]))),
        dtype=torch.float32,
    )
    payload["preview_feature_std"] = torch.as_tensor(
        payload.get("preview_feature_std", torch.ones(len(payload["preview_feature_names"]))),
        dtype=torch.float32,
    )
    payload["provenance_feature_mean"] = torch.as_tensor(
        payload.get("provenance_feature_mean", torch.zeros(len(payload["provenance_feature_names"]))),
        dtype=torch.float32,
    )
    payload["provenance_feature_std"] = torch.as_tensor(
        payload.get("provenance_feature_std", torch.ones(len(payload["provenance_feature_names"]))),
        dtype=torch.float32,
    )
    payload["actor_critic_reward_target"] = str(payload.get("actor_critic_reward_target", "immediate") or "immediate")
    payload["actor_critic_reward_mean"] = float(payload.get("actor_critic_reward_mean", 0.0))
    payload["actor_critic_reward_std"] = float(max(1.0e-6, _to_float(payload.get("actor_critic_reward_std", 1.0), 1.0)))
    payload["repair_slate_ranker_trained"] = bool(payload.get("repair_slate_ranker_trained", False))
    payload["repair_slate_ranker_target"] = str(payload.get("repair_slate_ranker_target", "") or "")
    payload["repair_slate_action_names"] = list(payload.get("repair_slate_action_names", []))
    payload["repair_tuple_ranker_trained"] = bool(payload.get("repair_tuple_ranker_trained", False))
    payload["repair_tuple_ranker_target"] = str(payload.get("repair_tuple_ranker_target", "") or "")
    payload["repair_tuple_action_names"] = list(payload.get("repair_tuple_action_names", []))
    payload["unified_candidate_ranker_trained"] = bool(payload.get("unified_candidate_ranker_trained", False))
    payload["unified_candidate_ranker_target"] = str(payload.get("unified_candidate_ranker_target", "") or "")
    payload["unified_candidate_route_tau"] = float(max(1.0e-6, _to_float(payload.get("unified_candidate_route_tau", 1.0), 1.0)))
    payload["unified_candidate_action_names"] = list(payload.get("unified_candidate_action_names", []))
    payload["repair_tuple_preview_value_target"] = str(payload.get("repair_tuple_preview_value_target", "") or "")
    payload["repair_tuple_child_value_lambda"] = float(max(0.0, _to_float(payload.get("repair_tuple_child_value_lambda", 0.0), 0.0)))
    payload["repair_tuple_regret_target"] = str(payload.get("repair_tuple_regret_target", "") or "")
    payload["repair_tuple_regret_weight"] = float(max(0.0, _to_float(payload.get("repair_tuple_regret_weight", 1.0), 1.0)))
    payload["mode_names"] = list(payload.get("mode_names", list(REPAIR_CRITIC_MODE_NAMES)))
    payload["path_relation_names"] = list(payload.get("path_relation_names", list(REPAIR_CRITIC_PATH_RELATION_NAMES)))
    payload["macro_head_trained"] = False
    payload["route_head_trained"] = False
    payload["q_head_trained"] = False
    payload["value_head_trained"] = False
    payload["regret_head_trained"] = False
    payload["path_head_trained"] = False
    payload["path_action_head_trained"] = False
    payload["path_relation_head_trained"] = False
    payload["path_mode_head_trained"] = False
    payload["path_improve_head_trained"] = False
    payload["model"] = model
    return payload


@torch.no_grad()
def predict_build_tuple_slate(
    bundle: dict[str, Any],
    row: Any,
) -> dict[str, Any]:
    out = {
        "trained": False,
        "best_index": None,
        "best_action": None,
        "best_child_key": None,
        "state_value_estimate": None,
        "rows": [],
    }
    if (
        not isinstance(bundle, dict)
        or "model" not in bundle
        or str(bundle.get("model_kind", "")) != REPAIR_CRITIC_BUILD_TUPLE_MODEL_KIND
        or not bool(bundle.get("build_tuple_ranker_trained", False))
    ):
        return out
    built = _build_build_tuple_rows([row], score_gap_floor=0.0)
    if not built:
        return out
    built_row = built[0]
    feature_names = list(bundle.get("feature_names", list(REPAIR_CRITIC_FEATURE_NAMES)))
    preview_feature_names = list(bundle.get("build_preview_feature_names", list(BUILD_TUPLE_PREVIEW_FEATURE_NAMES)))
    x = torch.tensor(
        [float(built_row["features"].get(name, 0.0)) for name in feature_names],
        dtype=torch.float32,
    ).unsqueeze(0)
    mean = torch.as_tensor(bundle.get("feature_mean", torch.zeros(len(feature_names))), dtype=torch.float32).reshape(1, -1)
    std = torch.as_tensor(bundle.get("feature_std", torch.ones(len(feature_names))), dtype=torch.float32).reshape(1, -1)
    preview_rows = [dict(preview_row) for preview_row in list(built_row.get("preview_rows", []) or []) if isinstance(preview_row, Mapping)]
    if not preview_rows:
        return out
    preview_x = torch.tensor(
        [build_preview_feature_vector(preview_row, feature_names=preview_feature_names) for preview_row in preview_rows],
        dtype=torch.float32,
    ).unsqueeze(0)
    preview_mean = torch.as_tensor(
        bundle.get("build_preview_feature_mean", torch.zeros(len(preview_feature_names))),
        dtype=torch.float32,
    ).reshape(1, 1, -1)
    preview_std = torch.as_tensor(
        bundle.get("build_preview_feature_std", torch.ones(len(preview_feature_names))),
        dtype=torch.float32,
    ).reshape(1, 1, -1)
    outputs = bundle["model"](
        _normalize_inputs(x, mean, std),
        _normalize_inputs(preview_x, preview_mean, preview_std),
    )
    scores = outputs["preview_score"].reshape(-1)
    rows_out: list[dict[str, Any]] = []
    for idx, preview_row in enumerate(preview_rows):
        row_out = dict(preview_row)
        row_out["utility_estimate"] = float(scores[idx].item())
        row_out["row_index"] = int(idx)
        rows_out.append(row_out)
    rows_out.sort(
        key=lambda row_out: (
            float(row_out.get("utility_estimate", float("-inf"))),
            -float(_to_float(row_out.get("slate_rank", 0.0), 0.0)),
            str(row_out.get("child_key", "")),
        ),
        reverse=True,
    )
    best_row = rows_out[0] if rows_out else None
    out.update({
        "trained": True,
        "best_index": None if best_row is None else int(best_row.get("row_index", 0)),
        "best_action": None if best_row is None else str(best_row.get("action", "")),
        "best_child_key": None if best_row is None else str(best_row.get("child_key", "")),
        "state_value_estimate": float(outputs["state_value"].reshape(-1)[0].item()),
        "rows": rows_out,
    })
    return out


@torch.no_grad()
def predict_unified_candidate_slate(
    bundle: dict[str, Any],
    row: Any,
    *,
    repair_preview_rows: Sequence[Mapping[str, Any]] | None = None,
    build_preview_rows: Sequence[Mapping[str, Any]] | None = None,
    repair_action_names: Sequence[str] = ("inv_steer", "repair_option"),
    build_action_names: Sequence[str] = ("replace", "wrap_un", "residual"),
) -> dict[str, Any]:
    out = {
        "trained": False,
        "best_index": None,
        "best_route": None,
        "best_action": None,
        "best_child_key": None,
        "state_value_estimate": None,
        "route_scores": {},
        "rows": [],
    }
    if (
        not isinstance(bundle, dict)
        or "model" not in bundle
        or str(bundle.get("model_kind", "")) not in {REPAIR_CRITIC_SHARED_MODEL_KIND, REPAIR_CRITIC_UNIFIED_CANDIDATE_MODEL_KIND}
        or not bool(bundle.get("unified_candidate_ranker_trained", False))
    ):
        return out

    feature_names = list(bundle.get("feature_names", list(REPAIR_CRITIC_FEATURE_NAMES)))
    path_feature_names = list(bundle.get("path_feature_names", list(REPAIR_CRITIC_PATH_FEATURE_NAMES)))
    preview_feature_names = list(bundle.get("preview_feature_names", list(UNIFIED_CANDIDATE_PREVIEW_FEATURE_NAMES)))
    provenance_feature_names = list(bundle.get("provenance_feature_names", list(preview_feature_names)))
    record = coerce_repair_feature_record(row)
    path_rows_seq = tuple(record.path_rows)
    if not path_rows_seq:
        return out

    allowed_repair_actions = {_normalize_action_name(v) for v in repair_action_names}
    allowed_build_actions = {_normalize_action_name(v) for v in build_action_names}
    base_row = coerce_repair_feature_row(row)
    repair_rows_list = _group_repair_preview_rows([
        shared_candidate_row_dict({**dict(pr), "route_source": "repair"}, route_source="repair")
        for pr in list(
            repair_preview_rows if repair_preview_rows is not None else list(base_row.get("inverse_repair_slate", []) or [])
        )
        if isinstance(pr, Mapping)
        and bool(pr.get("dedup_kept", True))
        and _normalize_action_name(pr.get("action", "")) in allowed_repair_actions
    ])
    build_rows_list = _group_build_preview_rows([
        shared_candidate_row_dict({**dict(pr), "route_source": "build"}, route_source="build")
        for pr in list(
            build_preview_rows if build_preview_rows is not None else list(base_row.get("controller_build_slate", []) or [])
        )
        if isinstance(pr, Mapping)
        and _normalize_action_name(pr.get("action", "")) in allowed_build_actions
    ])
    preview_rows_list = list(repair_rows_list) + list(build_rows_list)
    if not preview_rows_list:
        return out

    matched_rows: list[tuple[dict[str, Any], int]] = []
    for preview_row in preview_rows_list:
        path_idx = _match_candidate_path_index(path_rows_seq, preview_row)
        if path_idx is None:
            continue
        matched_rows.append((dict(preview_row), int(path_idx)))
    if not matched_rows:
        return out

    x = torch.tensor(
        repair_critic_feature_vector(record, feature_names=feature_names),
        dtype=torch.float32,
    ).unsqueeze(0)
    mean = torch.as_tensor(bundle.get("feature_mean", torch.zeros(len(feature_names))), dtype=torch.float32).reshape(1, -1)
    std = torch.as_tensor(bundle.get("feature_std", torch.ones(len(feature_names))), dtype=torch.float32).reshape(1, -1)
    x_n = _normalize_inputs(x, mean, std)

    path_x = torch.tensor(
        [repair_path_feature_vector(path_row, feature_names=path_feature_names) for path_row in path_rows_seq],
        dtype=torch.float32,
    ).unsqueeze(0)
    path_mean = torch.as_tensor(
        bundle.get("path_feature_mean", torch.zeros(len(path_feature_names))),
        dtype=torch.float32,
    ).reshape(1, 1, -1)
    path_std = torch.as_tensor(
        bundle.get("path_feature_std", torch.ones(len(path_feature_names))),
        dtype=torch.float32,
    ).reshape(1, 1, -1)
    path_x_n = _normalize_inputs(path_x, path_mean, path_std)
    path_mask = torch.ones((1, len(path_rows_seq)), dtype=torch.bool)

    preview_x = torch.tensor(
        [unified_candidate_preview_feature_vector(preview_row, feature_names=preview_feature_names) for preview_row, _ in matched_rows],
        dtype=torch.float32,
    ).unsqueeze(0)
    preview_mean = torch.as_tensor(
        bundle.get("preview_feature_mean", torch.zeros(len(preview_feature_names))),
        dtype=torch.float32,
    ).reshape(1, 1, -1)
    preview_std = torch.as_tensor(
        bundle.get("preview_feature_std", torch.ones(len(preview_feature_names))),
        dtype=torch.float32,
    ).reshape(1, 1, -1)
    preview_x_n = _normalize_inputs(preview_x, preview_mean, preview_std)
    preview_path_index = torch.tensor([[int(path_idx) for _, path_idx in matched_rows]], dtype=torch.long)
    preview_mask = torch.ones((1, len(matched_rows)), dtype=torch.bool)

    max_provenance = max(
        (len(list(preview_row.get("provenance_rows", []) or [preview_row])) for preview_row, _ in matched_rows),
        default=1,
    )
    provenance_x = torch.zeros((1, len(matched_rows), max_provenance, len(provenance_feature_names)), dtype=torch.float32)
    provenance_mask = torch.zeros((1, len(matched_rows), max_provenance), dtype=torch.bool)
    for j, (preview_row, _) in enumerate(matched_rows):
        provenance_rows = list(preview_row.get("provenance_rows", []) or [preview_row])
        for k in range(min(max_provenance, len(provenance_rows))):
            provenance_mask[0, j, k] = True
            provenance_x[0, j, k] = torch.tensor(
                unified_candidate_preview_feature_vector(provenance_rows[k], feature_names=provenance_feature_names),
                dtype=torch.float32,
            )
    provenance_mean = torch.as_tensor(
        bundle.get("provenance_feature_mean", torch.zeros(len(provenance_feature_names))),
        dtype=torch.float32,
    ).reshape(1, 1, 1, -1)
    provenance_std = torch.as_tensor(
        bundle.get("provenance_feature_std", torch.ones(len(provenance_feature_names))),
        dtype=torch.float32,
    ).reshape(1, 1, 1, -1)
    provenance_x_n = _normalize_inputs(provenance_x, provenance_mean, provenance_std)

    outputs = bundle["model"](
        x_n,
        path_x_n,
        path_mask,
        preview_x_n,
        preview_path_index,
        preview_mask,
        provenance_x_n,
        provenance_mask,
    )
    preview_scores = outputs["preview_utility"][0, : len(matched_rows)].reshape(-1)
    preview_regrets = outputs["preview_regret"][0, : len(matched_rows)].reshape(-1) if "preview_regret" in outputs else torch.zeros_like(preview_scores)
    state_value_estimate = float(outputs["value_pred"].reshape(-1)[0].item()) if "value_pred" in outputs else None
    path_probs = None
    if "path_logits" in outputs:
        path_probs = _masked_path_probs(outputs["path_logits"][:, : len(path_rows_seq)], path_mask).reshape(-1)
    regret_weight = float(max(0.0, _to_float(bundle.get("repair_tuple_regret_weight", 1.0), 1.0)))
    tau = float(max(1.0e-6, _to_float(bundle.get("unified_candidate_route_tau", 1.0), 1.0)))

    route_scores_raw: dict[str, list[float]] = {}
    rows_out: list[dict[str, Any]] = []
    best_key = None
    best_row = None
    for (preview_row, path_idx), score_t, regret_t in zip(matched_rows, preview_scores, preview_regrets):
        utility_est = float(score_t.item())
        regret_est = max(0.0, float(regret_t.item()))
        allocation_est = utility_est - (regret_weight * regret_est)
        route_name = str(preview_row.get("route_source", "") or "")
        route_scores_raw.setdefault(route_name, []).append(float(allocation_est))
        row_out = dict(preview_row)
        row_out.update({
            "path_index": int(path_idx),
            "matched_path": [int(v) for v in path_rows_seq[int(path_idx)].path],
            "utility_estimate": utility_est,
            "regret_estimate": regret_est,
            "allocation_estimate": allocation_est,
            "path_prob": float(path_probs[int(path_idx)].item()) if path_probs is not None else 0.0,
        })
        rows_out.append(row_out)
        key = (
            float(allocation_est),
            float(row_out["path_prob"]),
            str(route_name),
            str(row_out.get("child_key", "")),
        )
        if best_key is None or key > best_key:
            best_key = key
            best_row = row_out
    route_scores = {
        str(route_name): float(torch.logsumexp(torch.tensor(scores, dtype=torch.float32) / tau, dim=0).item())
        for route_name, scores in route_scores_raw.items()
    }
    rows_out.sort(
        key=lambda row_out: (
            float(row_out.get("allocation_estimate", row_out.get("utility_estimate", float("-inf")))),
            float(row_out.get("path_prob", 0.0)),
            str(row_out.get("route_source", "")),
        ),
        reverse=True,
    )
    best_route = None
    if route_scores:
        best_route = max(route_scores, key=lambda name: (route_scores[name], name))
    elif best_row is not None:
        best_route = str(best_row.get("route_source", "") or "")
    out.update({
        "trained": True,
        "best_index": None if best_row is None else int(rows_out.index(best_row)),
        "best_route": str(best_route or ""),
        "best_action": None if best_row is None else str(best_row.get("action", "")),
        "best_child_key": None if best_row is None else str(best_row.get("child_key", "")),
        "state_value_estimate": state_value_estimate,
        "route_scores": {str(k): float(v) for k, v in route_scores.items()},
        "rows": rows_out,
    })
    return out


def predict_shared_candidate_dual_slate(
    bundle: dict[str, Any],
    row: Any,
    *,
    repair_preview_rows: Sequence[Mapping[str, Any]] | None = None,
    build_preview_rows: Sequence[Mapping[str, Any]] | None = None,
    repair_action_names: Sequence[str] = ("inv_steer", "repair_option"),
    build_action_names: Sequence[str] = ("replace", "wrap_un", "residual"),
) -> dict[str, Any]:
    out = {
        "trained": False,
        "rows": [],
        "repair": {
            "best_action": None,
            "best_child_key": None,
            "best_score": None,
            "state_value_estimate": None,
        },
        "build": {
            "best_action": None,
            "best_child_key": None,
            "best_score": None,
            "state_value_estimate": None,
        },
        "common": {
            "best_route": None,
            "best_action": None,
            "best_child_key": None,
            "best_q": None,
            "state_value_estimate": None,
        },
    }
    if (
        not isinstance(bundle, dict)
        or "model" not in bundle
        or str(bundle.get("model_kind", "")) != REPAIR_CRITIC_SHARED_CANDIDATE_MODEL_KIND
        or not bool(bundle.get("shared_candidate_dual_trained", False))
    ):
        return out

    feature_names = list(bundle.get("feature_names", list(REPAIR_CRITIC_FEATURE_NAMES)))
    path_feature_names = list(bundle.get("path_feature_names", list(REPAIR_CRITIC_PATH_FEATURE_NAMES)))
    preview_feature_names = list(bundle.get("preview_feature_names", list(UNIFIED_CANDIDATE_PREVIEW_FEATURE_NAMES)))
    provenance_feature_names = list(bundle.get("provenance_feature_names", list(preview_feature_names)))
    record = coerce_repair_feature_record(row)
    path_rows_seq = tuple(record.path_rows)
    if not path_rows_seq:
        return out

    allowed_repair_actions = {_normalize_action_name(v) for v in repair_action_names}
    allowed_build_actions = {_normalize_action_name(v) for v in build_action_names}
    base_row = coerce_repair_feature_row(row)
    repair_rows_list = _group_repair_preview_rows([
        shared_candidate_row_dict({**dict(pr), "route_source": "repair"}, route_source="repair")
        for pr in list(
            repair_preview_rows if repair_preview_rows is not None else list(base_row.get("inverse_repair_slate", []) or [])
        )
        if isinstance(pr, Mapping)
        and bool(pr.get("dedup_kept", True))
        and _normalize_action_name(pr.get("action", "")) in allowed_repair_actions
    ])
    build_rows_list = _group_build_preview_rows([
        shared_candidate_row_dict({**dict(pr), "route_source": "build"}, route_source="build")
        for pr in list(
            build_preview_rows if build_preview_rows is not None else list(base_row.get("controller_build_slate", []) or [])
        )
        if isinstance(pr, Mapping)
        and _normalize_action_name(pr.get("action", "")) in allowed_build_actions
    ])
    preview_rows_list = list(repair_rows_list) + list(build_rows_list)
    if not preview_rows_list:
        return out

    matched_rows: list[tuple[dict[str, Any], int]] = []
    for preview_row in preview_rows_list:
        path_idx = _match_candidate_path_index(path_rows_seq, preview_row)
        if path_idx is None:
            continue
        matched_rows.append((dict(preview_row), int(path_idx)))
    if not matched_rows:
        return out

    x = torch.tensor(
        repair_critic_feature_vector(record, feature_names=feature_names),
        dtype=torch.float32,
    ).unsqueeze(0)
    mean = torch.as_tensor(bundle.get("feature_mean", torch.zeros(len(feature_names))), dtype=torch.float32).reshape(1, -1)
    std = torch.as_tensor(bundle.get("feature_std", torch.ones(len(feature_names))), dtype=torch.float32).reshape(1, -1)
    x_n = _normalize_inputs(x, mean, std)

    path_x = torch.tensor(
        [repair_path_feature_vector(path_row, feature_names=path_feature_names) for path_row in path_rows_seq],
        dtype=torch.float32,
    ).unsqueeze(0)
    path_mean = torch.as_tensor(
        bundle.get("path_feature_mean", torch.zeros(len(path_feature_names))),
        dtype=torch.float32,
    ).reshape(1, 1, -1)
    path_std = torch.as_tensor(
        bundle.get("path_feature_std", torch.ones(len(path_feature_names))),
        dtype=torch.float32,
    ).reshape(1, 1, -1)
    path_x_n = _normalize_inputs(path_x, path_mean, path_std)
    path_mask = torch.ones((1, len(path_rows_seq)), dtype=torch.bool)

    preview_x = torch.tensor(
        [unified_candidate_preview_feature_vector(preview_row, feature_names=preview_feature_names) for preview_row, _ in matched_rows],
        dtype=torch.float32,
    ).unsqueeze(0)
    preview_mean = torch.as_tensor(
        bundle.get("preview_feature_mean", torch.zeros(len(preview_feature_names))),
        dtype=torch.float32,
    ).reshape(1, 1, -1)
    preview_std = torch.as_tensor(
        bundle.get("preview_feature_std", torch.ones(len(preview_feature_names))),
        dtype=torch.float32,
    ).reshape(1, 1, -1)
    preview_x_n = _normalize_inputs(preview_x, preview_mean, preview_std)
    preview_path_index = torch.tensor([[int(path_idx) for _, path_idx in matched_rows]], dtype=torch.long)
    preview_mask = torch.ones((1, len(matched_rows)), dtype=torch.bool)

    max_provenance = max(
        (len(list(preview_row.get("provenance_rows", []) or [preview_row])) for preview_row, _ in matched_rows),
        default=1,
    )
    provenance_x = torch.zeros((1, len(matched_rows), max_provenance, len(provenance_feature_names)), dtype=torch.float32)
    provenance_mask = torch.zeros((1, len(matched_rows), max_provenance), dtype=torch.bool)
    for j, (preview_row, _) in enumerate(matched_rows):
        provenance_rows = list(preview_row.get("provenance_rows", []) or [preview_row])
        for k in range(min(max_provenance, len(provenance_rows))):
            provenance_mask[0, j, k] = True
            provenance_x[0, j, k] = torch.tensor(
                unified_candidate_preview_feature_vector(provenance_rows[k], feature_names=provenance_feature_names),
                dtype=torch.float32,
            )
    provenance_mean = torch.as_tensor(
        bundle.get("provenance_feature_mean", torch.zeros(len(provenance_feature_names))),
        dtype=torch.float32,
    ).reshape(1, 1, 1, -1)
    provenance_std = torch.as_tensor(
        bundle.get("provenance_feature_std", torch.ones(len(provenance_feature_names))),
        dtype=torch.float32,
    ).reshape(1, 1, 1, -1)
    provenance_x_n = _normalize_inputs(provenance_x, provenance_mean, provenance_std)

    outputs = bundle["model"](
        x_n,
        path_x_n,
        path_mask,
        preview_x_n,
        preview_path_index,
        preview_mask,
        provenance_x_n,
        provenance_mask,
    )
    repair_scores = outputs["repair_preview_score"][0, : len(matched_rows)].reshape(-1)
    build_scores = outputs["build_preview_score"][0, : len(matched_rows)].reshape(-1)
    common_utility = outputs["preview_utility"][0, : len(matched_rows)].reshape(-1)
    common_value = outputs["preview_value"][0, : len(matched_rows)].reshape(-1)
    common_regret = outputs["preview_regret"][0, : len(matched_rows)].reshape(-1)
    common_q = outputs["common_preview_q"][0, : len(matched_rows)].reshape(-1)
    repair_state_value = float(outputs["repair_state_value"].reshape(-1)[0].item()) if "repair_state_value" in outputs else None
    build_state_value = float(outputs["build_state_value"].reshape(-1)[0].item()) if "build_state_value" in outputs else None
    common_state_value = float(outputs["value_pred"].reshape(-1)[0].item()) if "value_pred" in outputs else None
    path_probs = None
    if "path_logits" in outputs:
        path_probs = _masked_path_probs(outputs["path_logits"][:, : len(path_rows_seq)], path_mask).reshape(-1)

    repair_best = None
    repair_best_key = None
    build_best = None
    build_best_key = None
    common_best = None
    common_best_key = None
    rows_out: list[dict[str, Any]] = []
    for idx, ((preview_row, path_idx), repair_score_t, build_score_t, utility_t, value_t, regret_t, q_t) in enumerate(
        zip(matched_rows, repair_scores, build_scores, common_utility, common_value, common_regret, common_q)
    ):
        route_name = str(preview_row.get("route_source", "") or "")
        route_score = float(repair_score_t.item()) if route_name == "repair" else float(build_score_t.item())
        row_out = dict(preview_row)
        row_out.update({
            "row_index": int(idx),
            "path_index": int(path_idx),
            "matched_path": [int(v) for v in path_rows_seq[int(path_idx)].path],
            "repair_score_estimate": float(repair_score_t.item()),
            "build_score_estimate": float(build_score_t.item()),
            "route_score_estimate": float(route_score),
            "common_utility_estimate": float(utility_t.item()),
            "common_value_estimate": float(value_t.item()),
            "common_regret_estimate": float(regret_t.item()),
            "common_q_estimate": float(q_t.item()),
            "path_prob": float(path_probs[int(path_idx)].item()) if path_probs is not None else 0.0,
        })
        rows_out.append(row_out)
        key = (
            float(route_score),
            float(row_out.get("path_prob", 0.0)),
            str(row_out.get("child_key", "")),
        )
        if route_name == "repair" and (repair_best_key is None or key > repair_best_key):
            repair_best_key = key
            repair_best = row_out
        if route_name == "build" and (build_best_key is None or key > build_best_key):
            build_best_key = key
            build_best = row_out
        common_key = (
            float(row_out.get("common_q_estimate", float("-inf"))),
            -float(row_out.get("common_regret_estimate", 0.0)),
            float(row_out.get("path_prob", 0.0)),
            str(row_out.get("child_key", "")),
        )
        if common_best_key is None or common_key > common_best_key:
            common_best_key = common_key
            common_best = row_out

    rows_out.sort(
        key=lambda row_out: (
            float(row_out.get("route_score_estimate", float("-inf"))),
            float(row_out.get("path_prob", 0.0)),
            str(row_out.get("route_source", "")),
        ),
        reverse=True,
    )
    out.update({
        "trained": True,
        "rows": rows_out,
        "repair": {
            "best_action": None if repair_best is None else str(repair_best.get("action", "")),
            "best_child_key": None if repair_best is None else str(repair_best.get("child_key", "")),
            "best_score": None if repair_best is None else float(repair_best.get("route_score_estimate", 0.0)),
            "state_value_estimate": repair_state_value,
        },
        "build": {
            "best_action": None if build_best is None else str(build_best.get("action", "")),
            "best_child_key": None if build_best is None else str(build_best.get("child_key", "")),
            "best_score": None if build_best is None else float(build_best.get("route_score_estimate", 0.0)),
            "state_value_estimate": build_state_value,
        },
        "common": {
            "best_route": None if common_best is None else str(common_best.get("route_source", "")),
            "best_action": None if common_best is None else str(common_best.get("action", "")),
            "best_child_key": None if common_best is None else str(common_best.get("child_key", "")),
            "best_q": None if common_best is None else float(common_best.get("common_q_estimate", 0.0)),
            "state_value_estimate": common_state_value,
        },
    })
    return out


def _predict_auxiliary(bundle: dict[str, Any], row: Any) -> dict[str, float]:
    feature_names = list(bundle.get("feature_names", list(REPAIR_CRITIC_FEATURE_NAMES)))
    head_names = list(bundle.get("head_names", list(REPAIR_CRITIC_HEAD_NAMES)))
    x = torch.tensor(repair_critic_feature_vector(row, feature_names=feature_names), dtype=torch.float32).unsqueeze(0)
    mean = torch.as_tensor(bundle.get("feature_mean", torch.zeros(len(feature_names))), dtype=torch.float32).reshape(1, -1)
    std = torch.as_tensor(bundle.get("feature_std", torch.ones(len(feature_names))), dtype=torch.float32).reshape(1, -1)
    logits = bundle["model"](_normalize_inputs(x, mean, std))
    if isinstance(logits, dict):
        logits = logits["aux_logits"]
    preds = _aux_predictions_from_logits(logits, head_names=head_names)
    return {name: float(preds[name].reshape(-1)[0].item()) for name in head_names}


@torch.no_grad()
def predict_repair_tuple_slate(
    bundle: dict[str, Any],
    row: Any,
    *,
    path_rows: Sequence[PathStateFeatures | Mapping[str, Any]] | None = None,
    preview_rows: Sequence[Mapping[str, Any]] | None = None,
    repair_action_names: Sequence[str] | None = None,
) -> dict[str, Any]:
    out = {
        "trained": False,
        "best_index": None,
        "best_path": None,
        "best_target_mode": None,
        "best_action": None,
        "best_child_key": None,
        "state_value_estimate": None,
        "child_value_lambda": 0.0,
        "regret_weight": 1.0,
        "rows": [],
    }
    if (
        not isinstance(bundle, dict)
        or "model" not in bundle
        or str(bundle.get("model_kind", "")) != REPAIR_CRITIC_SHARED_MODEL_KIND
        or not bool(bundle.get("repair_tuple_ranker_trained", False))
    ):
        return out

    feature_names = list(bundle.get("feature_names", list(REPAIR_CRITIC_FEATURE_NAMES)))
    path_feature_names = list(bundle.get("path_feature_names", list(REPAIR_CRITIC_PATH_FEATURE_NAMES)))
    preview_feature_names = list(bundle.get("preview_feature_names", list(REPAIR_CRITIC_PREVIEW_FEATURE_NAMES)))
    provenance_feature_names = list(bundle.get("provenance_feature_names", list(preview_feature_names)))
    record = coerce_repair_feature_record(row)
    path_rows_seq = tuple(
        pr if isinstance(pr, PathStateFeatures) else PathStateFeatures.from_row(pr)
        for pr in (path_rows if path_rows is not None else tuple(record.path_rows))
    )
    if not path_rows_seq or not preview_feature_names:
        return out

    allowed_actions = {
        _normalize_action_name(name)
        for name in (
            repair_action_names
            if repair_action_names is not None
            else list(bundle.get("repair_tuple_action_names", []) or [])
        )
        if _normalize_action_name(name)
    }
    preview_rows_list = _group_repair_preview_rows([
        dict(pr)
        for pr in list(preview_rows or [])
        if isinstance(pr, Mapping)
    ])
    matched_rows: list[tuple[dict[str, Any], int]] = []
    for preview_row in preview_rows_list:
        action_name = _normalize_action_name(preview_row.get("action", ""))
        if allowed_actions and action_name not in allowed_actions:
            continue
        path_idx = _match_path_mode_index(
            path_rows_seq,
            target_path=preview_row.get("path", ()),
            target_mode=preview_row.get("target_mode", None),
        )
        if path_idx is None:
            continue
        matched_rows.append((preview_row, int(path_idx)))
    if not matched_rows:
        return out

    x = torch.tensor(
        repair_critic_feature_vector(record, feature_names=feature_names),
        dtype=torch.float32,
    ).unsqueeze(0)
    mean = torch.as_tensor(bundle.get("feature_mean", torch.zeros(len(feature_names))), dtype=torch.float32).reshape(1, -1)
    std = torch.as_tensor(bundle.get("feature_std", torch.ones(len(feature_names))), dtype=torch.float32).reshape(1, -1)
    x_n = _normalize_inputs(x, mean, std)

    path_x = torch.tensor(
        [repair_path_feature_vector(path_row, feature_names=path_feature_names) for path_row in path_rows_seq],
        dtype=torch.float32,
    ).unsqueeze(0)
    path_mean = torch.as_tensor(
        bundle.get("path_feature_mean", torch.zeros(len(path_feature_names))),
        dtype=torch.float32,
    ).reshape(1, 1, -1)
    path_std = torch.as_tensor(
        bundle.get("path_feature_std", torch.ones(len(path_feature_names))),
        dtype=torch.float32,
    ).reshape(1, 1, -1)
    path_x_n = _normalize_inputs(path_x, path_mean, path_std)
    path_mask = torch.ones((1, len(path_rows_seq)), dtype=torch.bool)

    preview_x = torch.tensor(
        [repair_preview_feature_vector(preview_row, feature_names=preview_feature_names) for preview_row, _ in matched_rows],
        dtype=torch.float32,
    ).unsqueeze(0)
    preview_mean = torch.as_tensor(
        bundle.get("preview_feature_mean", torch.zeros(len(preview_feature_names))),
        dtype=torch.float32,
    ).reshape(1, 1, -1)
    preview_std = torch.as_tensor(
        bundle.get("preview_feature_std", torch.ones(len(preview_feature_names))),
        dtype=torch.float32,
    ).reshape(1, 1, -1)
    preview_x_n = _normalize_inputs(preview_x, preview_mean, preview_std)
    preview_path_index = torch.tensor([[int(path_idx) for _, path_idx in matched_rows]], dtype=torch.long)
    preview_mask = torch.ones((1, len(matched_rows)), dtype=torch.bool)
    max_provenance = max(
        (len(list(preview_row.get("provenance_rows", []) or [preview_row])) for preview_row, _ in matched_rows),
        default=1,
    )
    provenance_x = torch.zeros((1, len(matched_rows), max_provenance, len(provenance_feature_names)), dtype=torch.float32)
    provenance_mask = torch.zeros((1, len(matched_rows), max_provenance), dtype=torch.bool)
    for j, (preview_row, _) in enumerate(matched_rows):
        provenance_rows = list(preview_row.get("provenance_rows", []) or [preview_row])
        for k in range(min(max_provenance, len(provenance_rows))):
            provenance_mask[0, j, k] = True
            provenance_x[0, j, k] = torch.tensor(
                repair_preview_feature_vector(provenance_rows[k], feature_names=provenance_feature_names),
                dtype=torch.float32,
            )
    provenance_mean = torch.as_tensor(
        bundle.get("provenance_feature_mean", torch.zeros(len(provenance_feature_names))),
        dtype=torch.float32,
    ).reshape(1, 1, 1, -1)
    provenance_std = torch.as_tensor(
        bundle.get("provenance_feature_std", torch.ones(len(provenance_feature_names))),
        dtype=torch.float32,
    ).reshape(1, 1, 1, -1)
    provenance_x_n = _normalize_inputs(provenance_x, provenance_mean, provenance_std)

    outputs = bundle["model"](
        x_n,
        path_x_n,
        path_mask,
        preview_x_n,
        preview_path_index,
        preview_mask,
        provenance_x_n,
        provenance_mask,
    )
    preview_scores = outputs["preview_utility"][0, : len(matched_rows)].reshape(-1)
    preview_values = outputs["preview_value"][0, : len(matched_rows)].reshape(-1) if "preview_value" in outputs else preview_scores
    preview_regrets = outputs["preview_regret"][0, : len(matched_rows)].reshape(-1) if "preview_regret" in outputs else torch.zeros_like(preview_scores)
    child_value_target = str(bundle.get("repair_tuple_preview_value_target", "") or "")
    child_value_lambda = float(max(0.0, _to_float(bundle.get("repair_tuple_child_value_lambda", 0.0), 0.0)))
    regret_weight = float(max(0.0, _to_float(bundle.get("repair_tuple_regret_weight", 1.0), 1.0)))
    use_child_value = bool(child_value_target and child_value_target != "same_state_exact_child_log_gain" and child_value_lambda > 0.0)
    path_probs = None
    if "path_logits" in outputs:
        path_probs = _masked_path_probs(outputs["path_logits"][:, : len(path_rows_seq)], path_mask).reshape(-1)
    rows_out: list[dict[str, Any]] = []
    best_key = None
    best_idx = None
    for idx, ((preview_row, path_idx), score_t, value_t, regret_t) in enumerate(zip(matched_rows, preview_scores, preview_values, preview_regrets)):
        path_row = path_rows_seq[int(path_idx)]
        utility_est = float(score_t.item())
        value_est = float(value_t.item())
        regret_est = max(0.0, float(regret_t.item()))
        combined_est = utility_est + (child_value_lambda * max(0.0, value_est)) if use_child_value else utility_est
        allocation_est = combined_est - (regret_weight * regret_est)
        row_out = dict(preview_row)
        row_out.update({
            "path_index": int(path_idx),
            "matched_path": [int(v) for v in path_row.path],
            "matched_target_mode": str(path_row.target_mode),
            "utility_estimate": utility_est,
            "value_estimate": value_est,
            "regret_estimate": regret_est,
            "combined_estimate": combined_est,
            "allocation_estimate": allocation_est,
            "path_prob": float(path_probs[int(path_idx)].item()) if path_probs is not None else 0.0,
        })
        if "provenance_weights" in outputs:
            weights_t = outputs["provenance_weights"][0, idx, : max_provenance].reshape(-1)
            row_out["provenance_weights"] = [
                float(weights_t[k].item())
                for k in range(min(len(row_out.get("provenance_rows", []) or []), int(weights_t.shape[0])))
            ]
        rows_out.append(row_out)
        key = (
            float(row_out.get("allocation_estimate", row_out.get("combined_estimate", row_out["utility_estimate"]))),
            float(row_out["path_prob"]),
            -float(_to_float(row_out.get("beam_rank", 0.0), 0.0)),
            -float(_to_float(row_out.get("local_rank", 0.0), 0.0)),
            str(row_out.get("child_key", "")),
        )
        if best_key is None or key > best_key:
            best_key = key
            best_idx = idx
    rows_out.sort(
        key=lambda row_out: (
            float(row_out.get("allocation_estimate", row_out.get("combined_estimate", row_out.get("utility_estimate", float("-inf"))))),
            float(row_out.get("path_prob", 0.0)),
            -float(_to_float(row_out.get("beam_rank", 0.0), 0.0)),
            -float(_to_float(row_out.get("local_rank", 0.0), 0.0)),
        ),
        reverse=True,
    )
    state_value_estimate = None
    if "value_pred" in outputs:
        state_value_estimate = float(outputs["value_pred"].reshape(-1)[0].item())
    best_row = rows_out[0] if rows_out else None
    out.update({
        "trained": True,
        "best_index": None if best_idx is None else int(best_idx),
        "best_path": None if best_row is None else list(best_row.get("matched_path", best_row.get("path", []))),
        "best_target_mode": None if best_row is None else str(best_row.get("matched_target_mode", best_row.get("target_mode", ""))),
        "best_action": None if best_row is None else str(best_row.get("action", "")),
        "best_child_key": None if best_row is None else str(best_row.get("child_key", "")),
        "state_value_estimate": state_value_estimate,
        "child_value_lambda": float(child_value_lambda if use_child_value else 0.0),
        "regret_weight": float(regret_weight),
        "rows": rows_out,
    })
    return out


@torch.no_grad()
def predict_repair_build_route(
    bundle: dict[str, Any],
    row: Any,
    *,
    repair_tuple_bundle: dict[str, Any] | None = None,
    build_tuple_bundle: dict[str, Any] | None = None,
) -> dict[str, Any]:
    out = {
        "trained": False,
        "best_route": None,
        "repair_prob": None,
        "build_prob": None,
        "margin_estimate": None,
        "repair_summary": None,
        "build_summary": None,
    }
    if (
        not isinstance(bundle, dict)
        or "model" not in bundle
        or str(bundle.get("model_kind", "")) != REPAIR_CRITIC_ROUTE_COMPARE_MODEL_KIND
        or not bool(bundle.get("repair_build_route_compare_trained", False))
    ):
        return out
    built = _build_repair_build_route_rows(
        [row],
        margin_floor=0.0,
        repair_tuple_bundle=repair_tuple_bundle,
        build_tuple_bundle=build_tuple_bundle,
    )
    if not built:
        return out
    built_row = built[0]
    feature_names = list(bundle.get("feature_names", list(tuple(REPAIR_CRITIC_FEATURE_NAMES) + tuple(REPAIR_ROUTE_COMPARE_EXTRA_FEATURE_NAMES))))
    x = torch.tensor(
        [float(built_row["features"].get(name, 0.0)) for name in feature_names],
        dtype=torch.float32,
    ).unsqueeze(0)
    mean = torch.as_tensor(bundle.get("feature_mean", torch.zeros(len(feature_names))), dtype=torch.float32).reshape(1, -1)
    std = torch.as_tensor(bundle.get("feature_std", torch.ones(len(feature_names))), dtype=torch.float32).reshape(1, -1)
    outputs = bundle["model"](_normalize_inputs(x, mean, std))
    repair_prob = float(torch.sigmoid(outputs["route_logit"].reshape(-1)[0]).item())
    margin_estimate = float(outputs["margin_pred"].reshape(-1)[0].item())
    out.update({
        "trained": True,
        "best_route": "repair" if repair_prob >= 0.5 else "build",
        "repair_prob": float(repair_prob),
        "build_prob": float(1.0 - repair_prob),
        "margin_estimate": margin_estimate,
        "repair_summary": dict(built_row.get("repair_summary", {}) or {}),
        "build_summary": dict(built_row.get("build_summary", {}) or {}),
        "exact_margin": float(built_row.get("route_margin", 0.0) or 0.0),
    })
    return out


@torch.no_grad()
def predict_repair_controller_heads(bundle: dict[str, Any], row: Any) -> dict[str, Any]:
    if not isinstance(bundle, dict) or "model" not in bundle:
        raise ValueError("Loaded repair critic bundle is missing a model.")
    aux = _predict_auxiliary(bundle, row)
    out = {
        "auxiliary": aux,
        "macro_action": {
            "trained": False,
            "best_action": None,
            "probs": {},
        },
        "route": {
            "trained": False,
            "best_route": None,
            "probs": {},
        },
        "action_value": {
            "trained": False,
            "best_action": None,
            "estimates": {},
            "normalized_estimates": {},
        },
        "value": {
            "trained": False,
            "estimate": None,
            "normalized_estimate": None,
        },
        "path": {
            "trained": False,
            "best_path": None,
            "best_target_mode": None,
            "rows": [],
        },
        "path_action": {
            "trained": False,
            "best_path": None,
            "best_route": None,
            "best_action": None,
            "rows": [],
            "tuples": [],
        },
    }
    if str(bundle.get("model_kind", "")) != REPAIR_CRITIC_SHARED_MODEL_KIND:
        return out

    feature_names = list(bundle.get("feature_names", list(REPAIR_CRITIC_FEATURE_NAMES)))
    path_feature_names = list(bundle.get("path_feature_names", list(REPAIR_CRITIC_PATH_FEATURE_NAMES)))
    macro_action_names = list(bundle.get("macro_action_names", list(REPAIR_CRITIC_MACRO_ACTION_NAMES)))
    route_names = list(bundle.get("route_names", list(REPAIR_CRITIC_ROUTE_NAMES)))
    mode_names = list(bundle.get("mode_names", list(REPAIR_CRITIC_MODE_NAMES)))
    relation_names = list(bundle.get("path_relation_names", list(REPAIR_CRITIC_PATH_RELATION_NAMES)))
    record = coerce_repair_feature_record(row)
    x = torch.tensor(repair_critic_feature_vector(record, feature_names=feature_names), dtype=torch.float32).unsqueeze(0)
    mean = torch.as_tensor(bundle.get("feature_mean", torch.zeros(len(feature_names))), dtype=torch.float32).reshape(1, -1)
    std = torch.as_tensor(bundle.get("feature_std", torch.ones(len(feature_names))), dtype=torch.float32).reshape(1, -1)
    x_n = _normalize_inputs(x, mean, std)

    path_rows = tuple(record.path_rows)
    if path_rows:
        path_x = torch.tensor(
            [repair_path_feature_vector(path_row, feature_names=path_feature_names) for path_row in path_rows],
            dtype=torch.float32,
        ).unsqueeze(0)
        path_mean = torch.as_tensor(
            bundle.get("path_feature_mean", torch.zeros(len(path_feature_names))),
            dtype=torch.float32,
        ).reshape(1, 1, -1)
        path_std = torch.as_tensor(
            bundle.get("path_feature_std", torch.ones(len(path_feature_names))),
            dtype=torch.float32,
        ).reshape(1, 1, -1)
        path_x_n = _normalize_inputs(path_x, path_mean, path_std)
        valid_mask = torch.ones((1, len(path_rows)), dtype=torch.bool)
        outputs = bundle["model"](x_n, path_x_n, valid_mask)
    else:
        outputs = bundle["model"](x_n, None)

    route_probs_map: dict[str, float] = {}
    best_route_name: str | None = None
    if bool(bundle.get("route_head_trained", False)) and route_names:
        route_probs = torch.softmax(outputs["route_logits"], dim=-1).reshape(-1)
        best_idx = int(torch.argmax(route_probs).item())
        best_route_name = str(route_names[best_idx])
        route_probs_map = {
            str(name): float(route_probs[i].item())
            for i, name in enumerate(route_names)
        }
        out["route"] = {
            "trained": True,
            "best_route": best_route_name,
            "probs": route_probs_map,
        }
    if bool(bundle.get("macro_head_trained", False)) and macro_action_names:
        macro_probs_raw = torch.softmax(outputs["macro_logits"], dim=-1).reshape(-1)
        macro_probs_map = {
            str(name): float(macro_probs_raw[i].item())
            for i, name in enumerate(macro_action_names)
        }
        if route_probs_map:
            hier_probs_map = {str(name): 0.0 for name in macro_action_names}
            for route_name, route_prob in route_probs_map.items():
                route_actions = [
                    str(name)
                    for name in macro_action_names
                    if REPAIR_CRITIC_ACTION_ROUTE_MAP.get(str(name), "") == str(route_name)
                ]
                if not route_actions:
                    continue
                route_mass = sum(float(macro_probs_map.get(name, 0.0)) for name in route_actions)
                if route_mass <= 1.0e-12:
                    continue
                for name in route_actions:
                    hier_probs_map[name] = float(route_prob * macro_probs_map[name] / route_mass)
            total = sum(hier_probs_map.values())
            if total > 1.0e-12:
                macro_probs_map = {name: float(value / total) for name, value in hier_probs_map.items()}
        best_action = max(macro_probs_map, key=lambda name: (macro_probs_map.get(name, 0.0), name))
        out["macro_action"] = {
            "trained": True,
            "best_action": str(best_action),
            "probs": macro_probs_map,
            "best_action_route": best_route_name,
        }
    if bool(bundle.get("q_head_trained", False)) and macro_action_names and ("q_values" in outputs):
        reward_mean = float(bundle.get("actor_critic_reward_mean", 0.0))
        reward_std = float(max(1.0e-6, _to_float(bundle.get("actor_critic_reward_std", 1.0), 1.0)))
        q_norm = outputs["q_values"].reshape(-1)
        q_est = q_norm * reward_std + reward_mean
        best_idx = int(torch.argmax(q_est).item())
        out["action_value"] = {
            "trained": True,
            "best_action": str(macro_action_names[best_idx]),
            "estimates": {
                str(name): float(q_est[i].item())
                for i, name in enumerate(macro_action_names)
            },
            "normalized_estimates": {
                str(name): float(q_norm[i].item())
                for i, name in enumerate(macro_action_names)
            },
        }
    if bool(bundle.get("value_head_trained", False)) and ("value_pred" in outputs):
        reward_mean = float(bundle.get("actor_critic_reward_mean", 0.0))
        reward_std = float(max(1.0e-6, _to_float(bundle.get("actor_critic_reward_std", 1.0), 1.0)))
        value_norm = float(outputs["value_pred"].reshape(-1)[0].item())
        out["value"] = {
            "trained": True,
            "estimate": float(value_norm * reward_std + reward_mean),
            "normalized_estimate": float(value_norm),
        }

    if bool(bundle.get("path_head_trained", False)) and path_rows and ("path_logits" in outputs):
        valid_mask = torch.ones((1, len(path_rows)), dtype=torch.bool)
        path_probs = _masked_path_probs(outputs["path_logits"][:, :len(path_rows)], valid_mask).reshape(-1)
        best_idx = int(torch.argmax(path_probs).item())
        policy_weights = None
        if "path_policy_weights" in outputs:
            policy_weights = outputs["path_policy_weights"][0, :len(path_rows)].reshape(-1)
        rows_out = []
        for i, path_row in enumerate(path_rows):
            row_out = {
                "path": [int(v) for v in path_row.path],
                "target_mode": str(path_row.target_mode),
                "score": float(outputs["path_logits"][0, i].item()),
                "prob": float(path_probs[i].item()),
                "weighted_rel_gain": float(path_row.weighted_rel_gain),
            }
            if policy_weights is not None:
                row_out["policy_weight"] = float(policy_weights[i].item())
            if bool(bundle.get("path_relation_head_trained", False)) and relation_names:
                rel_probs = torch.softmax(outputs["path_relation_logits"][0, i], dim=-1).reshape(-1)
                rel_idx = int(torch.argmax(rel_probs).item())
                row_out["best_relation"] = str(relation_names[rel_idx])
                row_out["relation_probs"] = {
                    str(name): float(rel_probs[j].item())
                    for j, name in enumerate(relation_names)
                }
            if bool(bundle.get("path_mode_head_trained", False)) and mode_names:
                mode_probs = torch.softmax(outputs["path_mode_logits"][0, i], dim=-1).reshape(-1)
                mode_idx = int(torch.argmax(mode_probs).item())
                row_out["best_mode"] = str(mode_names[mode_idx])
                row_out["mode_probs"] = {
                    str(name): float(mode_probs[j].item())
                    for j, name in enumerate(mode_names)
                }
            if bool(bundle.get("path_improve_head_trained", False)):
                row_out["improvement_estimate"] = float(torch.sigmoid(outputs["path_improve"][0, i]).item())
            rows_out.append(row_out)
        out["path"] = {
            "trained": True,
            "best_path": list(rows_out[best_idx]["path"]),
            "best_target_mode": str(rows_out[best_idx].get("best_mode", rows_out[best_idx]["target_mode"])),
            "rows": rows_out,
        }
    if (
        bool(bundle.get("path_action_head_trained", False))
        and path_rows
        and ("path_route_logits" in outputs)
        and ("path_macro_logits" in outputs)
        and ("path_q_values" in outputs)
    ):
        repair_slate_ranker = bool(bundle.get("repair_slate_ranker_trained", False))
        reward_mean = 0.0 if repair_slate_ranker else float(bundle.get("actor_critic_reward_mean", 0.0))
        reward_std = 1.0 if repair_slate_ranker else float(max(1.0e-6, _to_float(bundle.get("actor_critic_reward_std", 1.0), 1.0)))
        allowed_repair_actions = {
            _normalize_action_name(name)
            for name in list(bundle.get("repair_slate_action_names", []) or [])
            if _normalize_action_name(name)
        }
        if repair_slate_ranker and not allowed_repair_actions:
            allowed_repair_actions = set(_REPAIR_CRITIC_DEFAULT_REPAIR_ACTION_SET)
        path_valid = torch.ones((1, len(path_rows)), dtype=torch.bool)
        if "path_logits" in outputs:
            path_probs_tensor = _masked_path_probs(outputs["path_logits"][:, :len(path_rows)], path_valid).reshape(-1)
        else:
            path_probs_tensor = torch.full((len(path_rows),), 1.0 / float(max(1, len(path_rows))), dtype=torch.float32)
        rows_out: list[dict[str, Any]] = []
        tuple_rows: list[dict[str, Any]] = []
        best_q_estimates: dict[str, float] = {str(name): float("-inf") for name in macro_action_names}
        best_q_normalized: dict[str, float] = {str(name): float("-inf") for name in macro_action_names}
        best_tuple_key = None
        best_tuple_row: dict[str, Any] | None = None
        for i, path_row in enumerate(path_rows):
            route_probs = torch.softmax(outputs["path_route_logits"][0, i], dim=-1).reshape(-1)
            macro_probs_raw = torch.softmax(outputs["path_macro_logits"][0, i], dim=-1).reshape(-1)
            q_norm = outputs["path_q_values"][0, i].reshape(-1)
            q_est = q_norm * reward_std + reward_mean
            route_probs_map = {
                str(name): float(route_probs[j].item())
                for j, name in enumerate(route_names)
            }
            macro_probs_map = {
                str(name): float(macro_probs_raw[j].item())
                for j, name in enumerate(macro_action_names)
            }
            hier_probs_map = {str(name): 0.0 for name in macro_action_names}
            within_route_map = {str(name): 0.0 for name in macro_action_names}
            if repair_slate_ranker:
                route_probs_map = {"repair": 1.0}
                allowed_action_list = [
                    str(name)
                    for name in macro_action_names
                    if _normalize_action_name(name) in allowed_repair_actions
                ]
                if not allowed_action_list:
                    allowed_action_list = list(REPAIR_CRITIC_DEFAULT_REPAIR_ACTION_SET)  # noqa: F821
                allowed_idx = [macro_action_names.index(name) for name in allowed_action_list if name in macro_action_names]
                if allowed_idx:
                    allowed_logits = q_norm[allowed_idx]
                    allowed_probs = torch.softmax(allowed_logits, dim=-1).reshape(-1)
                    for local_i, action_name in enumerate(allowed_action_list):
                        prob = float(allowed_probs[local_i].item())
                        within_route_map[action_name] = prob
                        hier_probs_map[action_name] = prob
                row_best_action = max(
                    allowed_action_list,
                    key=lambda name: (float(q_est[macro_action_names.index(name)].item()), name),
                )
            else:
                for route_name, route_prob in route_probs_map.items():
                    route_actions = [
                        str(name)
                        for name in macro_action_names
                        if REPAIR_CRITIC_ACTION_ROUTE_MAP.get(str(name), "") == str(route_name)
                    ]
                    if not route_actions:
                        continue
                    route_mass = sum(float(macro_probs_map.get(name, 0.0)) for name in route_actions)
                    if route_mass <= 1.0e-12:
                        continue
                    for name in route_actions:
                        local_prob = float(macro_probs_map.get(name, 0.0)) / route_mass
                        within_route_map[name] = float(local_prob)
                        hier_probs_map[name] = float(route_prob * local_prob)
                row_best_action = max(hier_probs_map, key=lambda name: (hier_probs_map.get(name, 0.0), name))
            row_best_route = str(REPAIR_CRITIC_ACTION_ROUTE_MAP.get(row_best_action, "") or "")
            row_out = {
                "path": [int(v) for v in path_row.path],
                "target_mode": str(path_row.target_mode),
                "path_prob": float(path_probs_tensor[i].item()),
                "best_route": row_best_route,
                "best_action": str(row_best_action),
                "route_probs": route_probs_map,
                "action_probs": hier_probs_map,
                "within_route_action_probs": within_route_map,
                "q_estimates": {
                    str(name): float(q_est[j].item())
                    for j, name in enumerate(macro_action_names)
                },
                "normalized_estimates": {
                    str(name): float(q_norm[j].item())
                    for j, name in enumerate(macro_action_names)
                },
            }
            rows_out.append(row_out)
            for action_name in macro_action_names:
                if repair_slate_ranker and _normalize_action_name(action_name) not in allowed_repair_actions:
                    continue
                route_name = str(REPAIR_CRITIC_ACTION_ROUTE_MAP.get(str(action_name), "") or "")
                tuple_row = {
                    "path": [int(v) for v in path_row.path],
                    "target_mode": str(path_row.target_mode),
                    "route": route_name,
                    "action": str(action_name),
                    "path_prob": float(path_probs_tensor[i].item()),
                    "route_prob": float(route_probs_map.get(route_name, 0.0)),
                    "action_prob": float(hier_probs_map.get(str(action_name), 0.0)),
                    "within_route_action_prob": float(within_route_map.get(str(action_name), 0.0)),
                    "q_estimate": float(row_out["q_estimates"][str(action_name)]),
                    "q_normalized": float(row_out["normalized_estimates"][str(action_name)]),
                }
                tuple_rows.append(tuple_row)
                best_q_estimates[str(action_name)] = max(best_q_estimates.get(str(action_name), float("-inf")), float(tuple_row["q_estimate"]))
                best_q_normalized[str(action_name)] = max(best_q_normalized.get(str(action_name), float("-inf")), float(tuple_row["q_normalized"]))
                if repair_slate_ranker:
                    key = (
                        float(tuple_row["q_estimate"]),
                        float(tuple_row["path_prob"]),
                        float(tuple_row["action_prob"]),
                        tuple(tuple_row["path"]),
                        str(action_name),
                    )
                else:
                    key = (
                        float(tuple_row["action_prob"]),
                        float(tuple_row["q_estimate"]),
                        tuple(tuple_row["path"]),
                        str(action_name),
                    )
                if best_tuple_key is None or key > best_tuple_key:
                    best_tuple_key = key
                    best_tuple_row = tuple_row
        out["path_action"] = {
            "trained": True,
            "best_path": None if best_tuple_row is None else list(best_tuple_row["path"]),
            "best_route": None if best_tuple_row is None else str(best_tuple_row["route"]),
            "best_action": None if best_tuple_row is None else str(best_tuple_row["action"]),
            "rows": rows_out,
            "tuples": tuple_rows,
        }
        finite_q = {
            name: value
            for name, value in best_q_estimates.items()
            if math.isfinite(float(value))
        }
        finite_q_norm = {
            name: value
            for name, value in best_q_normalized.items()
            if math.isfinite(float(value))
        }
        if finite_q:
            best_action = max(finite_q, key=lambda name: (finite_q.get(name, float("-inf")), name))
            out["action_value"] = {
                "trained": True,
                "best_action": str(best_action),
                "estimates": {str(name): float(value) for name, value in finite_q.items()},
                "normalized_estimates": {
                    str(name): float(finite_q_norm.get(name, 0.0))
                    for name in finite_q
                },
            }
    return out


@torch.no_grad()
def predict_repair_critic(bundle: dict[str, Any], row: Any) -> dict[str, float]:
    return dict(predict_repair_controller_heads(bundle, row).get("auxiliary", {}))


def iter_inverse_experiment_rows(payload: Any) -> Iterable[dict[str, Any]]:
    if isinstance(payload, dict):
        for key in ("inverse_experiment_log", "inverse_log"):
            inv_log = payload.get(key, None)
            if not isinstance(inv_log, list):
                continue
            for row in inv_log:
                if isinstance(row, dict):
                    yield row
            return
        if "status" in payload and "parent_expr" in payload:
            yield payload
            return
        for value in payload.values():
            yield from iter_inverse_experiment_rows(value)
        return
    if isinstance(payload, list):
        for item in payload:
            yield from iter_inverse_experiment_rows(item)


def load_inverse_experiment_rows(paths: Sequence[str | Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        rows.extend(dict(row) for row in iter_inverse_experiment_rows(payload))
    return rows
