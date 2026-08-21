# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""Compatibility facade for the legacy repair-critic implementation."""

from __future__ import annotations

import inspect
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from nestynet_sr.sr_search.factorized_search.engine.signals import PathStateFeatures
from nestynet_sr.sr_search.factorized_search.policy.features import (
    coerce_repair_feature_record,
    coerce_repair_feature_row,
)
from nestynet_sr.sr_search.factorized_search.shared_candidate import (
    SHARED_CANDIDATE_MASK_FIELD_NAMES,
    coerce_shared_candidate_record,
    shared_candidate_row_dict,
)

from . import _repair_critic_features as _features
from . import _repair_critic_models as _models
from . import _repair_critic_prediction as _prediction
from . import _repair_critic_training as _training

# Retain the original imported module globals for deferred annotations and for
# callers that historically inspected this legacy module directly.
_LEGACY_IMPORTED_GLOBALS = (
    Any,
    F,
    Iterable,
    Mapping,
    Path,
    PathStateFeatures,
    SHARED_CANDIDATE_MASK_FIELD_NAMES,
    Sequence,
    coerce_repair_feature_record,
    coerce_repair_feature_row,
    coerce_shared_candidate_record,
    json,
    math,
    nn,
    shared_candidate_row_dict,
    torch,
)

_CONSTANT_NAMES = (
    "REPAIR_CRITIC_FEATURE_NAMES",
    "REPAIR_CRITIC_HEAD_NAMES",
    "REPAIR_CRITIC_MACRO_ACTION_NAMES",
    "REPAIR_CRITIC_ROUTE_NAMES",
    "REPAIR_CRITIC_ACTION_ROUTE_MAP",
    "REPAIR_CRITIC_PATH_FEATURE_NAMES",
    "REPAIR_CRITIC_PREVIEW_FEATURE_NAMES",
    "BUILD_TUPLE_PREVIEW_FEATURE_NAMES",
    "UNIFIED_CANDIDATE_PREVIEW_FEATURE_NAMES",
    "REPAIR_ROUTE_COMPARE_EXTRA_FEATURE_NAMES",
    "REPAIR_CRITIC_PATH_RELATION_NAMES",
    "REPAIR_CRITIC_MODE_NAMES",
    "REPAIR_CRITIC_SHARED_MODEL_KIND",
    "REPAIR_CRITIC_LEGACY_MODEL_KIND",
    "REPAIR_CRITIC_ROUTE_COMPARE_MODEL_KIND",
    "REPAIR_CRITIC_BUILD_TUPLE_MODEL_KIND",
    "REPAIR_CRITIC_UNIFIED_CANDIDATE_MODEL_KIND",
    "REPAIR_CRITIC_SHARED_CANDIDATE_MODEL_KIND",
    "REPAIR_CRITIC_DEFAULT_UTILITY_WEIGHTS",
    "_REPAIR_CRITIC_NEGATIVE_TERMINAL_STATUSES",
    "_REPAIR_CRITIC_DEFAULT_HEAD_SET",
    "_REPAIR_CRITIC_DEFAULT_MACRO_ACTION_SET",
    "_REPAIR_CRITIC_DEFAULT_ROUTE_SET",
    "_REPAIR_CRITIC_DEFAULT_RELATION_SET",
    "_REPAIR_CRITIC_DEFAULT_MODE_SET",
    "_REPAIR_CRITIC_DEFAULT_REPAIR_ACTION_SET",
)

_FEATURE_FUNCTIONS = (
    "_to_float",
    "_clamp01",
    "_finite_float_or_none",
    "_row_first",
    "_safe_log1p",
    "_safe_neg_log10",
    "_normalize_action_name",
    "_route_name_for_action",
    "_normalize_mode_name",
    "_normalize_mapping_kind",
    "_mapping_bucket",
    "_normalize_relation_name",
    "extract_repair_critic_features",
    "extract_repair_path_features",
    "repair_critic_feature_vector",
    "repair_path_feature_vector",
    "_preview_root_category",
    "_path_source_bucket",
    "repair_preview_feature_vector",
    "build_preview_feature_vector",
    "unified_candidate_preview_feature_vector",
    "_row_has_supervised_outcome",
    "estimate_reward_per_s_scale",
    "make_repair_critic_target",
    "collect_repair_critic_examples",
    "_extract_macro_action_label",
    "_extract_route_label",
    "_match_selected_path_index",
    "_build_training_rows",
    "_build_oracle_pretrain_rows",
    "_resolve_actor_critic_reward",
    "_common_candidate_q_target",
    "_match_path_mode_index",
    "_match_candidate_path_index",
    "_make_repair_slate_utility",
    "_group_repair_preview_rows",
    "_group_build_preview_rows",
    "_summarize_route_candidate_rows",
    "_summarize_learned_route_prediction",
    "_build_repair_build_route_feature_dict",
    "_build_repair_build_route_rows",
    "_build_build_tuple_rows",
    "_build_repair_slate_rows",
    "_build_repair_tuple_rows",
    "_build_unified_candidate_rows",
    "_build_actor_critic_rows",
)

_MODEL_CLASSES = (
    "_RepairCriticNet",
    "_RepairRouteCompareNet",
    "_BuildTupleRankerNet",
    "_RepairControllerSharedNet",
    "_SharedCandidateDualRankerNet",
)

_MODEL_FUNCTIONS = (
    "_binary_head_pos_weight",
    "_split_indices",
    "_aux_predictions_from_logits",
    "_metrics_from_preds",
    "_aux_loss_from_logits",
    "_macro_metrics_from_logits",
    "_inverse_frequency_class_weights",
    "_route_action_mask",
    "_route_masked_action_logits",
    "_hierarchical_macro_metrics",
    "_masked_path_cross_entropy",
    "_masked_path_probs",
    "_gather_path_head",
    "_path_metrics_from_logits",
    "_flat_classification_metrics",
    "_binary_classification_metrics",
    "_regression_metrics",
    "_gather_path_action_scores",
    "_masked_log_softmax",
    "_listwise_slate_loss",
    "_pairwise_rank_loss",
    "_masked_pairwise_pairs_from_targets",
    "_slate_rank_metrics",
    "_route_emergence_metrics",
    "_compute_feature_stats",
    "_normalize_inputs",
    "_maybe_init_model_from_bundle",
    "_bundle_kind_from_payload",
)

_TRAINING_FUNCTIONS = (
    "train_build_tuple_ranker",
    "train_repair_critic",
    "pretrain_repair_controller_from_oracle_tasks",
    "train_repair_controller_actor_critic",
    "train_repair_controller_slate_ranker",
    "train_repair_controller_tuple_ranker",
    "train_unified_candidate_ranker",
    "train_shared_candidate_dual_ranker",
    "train_repair_build_route_comparator",
)

_PREDICTION_FUNCTIONS = (
    "save_repair_critic_bundle",
    "load_repair_critic_bundle",
    "predict_build_tuple_slate",
    "predict_unified_candidate_slate",
    "predict_shared_candidate_dual_slate",
    "_predict_auxiliary",
    "predict_repair_tuple_slate",
    "predict_repair_build_route",
    "predict_repair_controller_heads",
    "predict_repair_critic",
    "iter_inverse_experiment_rows",
    "load_inverse_experiment_rows",
)

_IMPLEMENTATION_MODULES = (_features, _models, _training, _prediction)


for _name in _CONSTANT_NAMES:
    globals()[_name] = getattr(_features, _name)

for _module, _names in (
    (_features, _FEATURE_FUNCTIONS),
    (_models, _MODEL_FUNCTIONS),
    (_training, _TRAINING_FUNCTIONS),
    (_prediction, _PREDICTION_FUNCTIONS),
):
    for _name in _names:
        _function = getattr(_module, _name)
        _function.__module__ = __name__
        _function.__qualname__ = _name
        globals()[_name] = _function

for _name in _MODEL_CLASSES:
    _class = getattr(_models, _name)
    _class.__module__ = __name__
    for _member in _class.__dict__.values():
        if inspect.isfunction(_member):
            _member.__module__ = __name__
    globals()[_name] = _class

_features.predict_build_tuple_slate = globals()["predict_build_tuple_slate"]
_features.predict_repair_tuple_slate = globals()["predict_repair_tuple_slate"]

del _class, _function, _member, _module, _name, _names
