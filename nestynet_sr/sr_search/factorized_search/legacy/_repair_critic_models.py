# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

from __future__ import annotations

"""Torch models, losses, and metrics for repair-critic training."""

from typing import Any, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from ._repair_critic_features import (
    REPAIR_CRITIC_ACTION_ROUTE_MAP,
    REPAIR_CRITIC_BUILD_TUPLE_MODEL_KIND,
    REPAIR_CRITIC_HEAD_NAMES,
    REPAIR_CRITIC_LEGACY_MODEL_KIND,
    REPAIR_CRITIC_MODE_NAMES,
    REPAIR_CRITIC_PATH_RELATION_NAMES,
    REPAIR_CRITIC_ROUTE_COMPARE_MODEL_KIND,
    REPAIR_CRITIC_SHARED_CANDIDATE_MODEL_KIND,
    REPAIR_CRITIC_SHARED_MODEL_KIND,
    REPAIR_CRITIC_UNIFIED_CANDIDATE_MODEL_KIND,
)

class _RepairCriticNet(nn.Module):
    """Legacy aux-only network kept for checkpoint compatibility."""

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        hidden_dim = max(4, int(hidden_dim))
        self.lin1 = nn.Linear(int(input_dim), hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, hidden_dim)
        self.head = nn.Linear(hidden_dim, len(REPAIR_CRITIC_HEAD_NAMES))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.lin1(x))
        h = F.silu(self.lin2(h))
        return self.head(h)


class _RepairRouteCompareNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        hidden_dim = max(8, int(hidden_dim))
        self.lin1 = nn.Linear(int(input_dim), hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, hidden_dim)
        self.route_logit = nn.Linear(hidden_dim, 1)
        self.margin_head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        h = F.silu(self.lin1(x))
        h = F.silu(self.lin2(h))
        return {
            "route_logit": self.route_logit(h).squeeze(-1),
            "margin_pred": self.margin_head(h).squeeze(-1),
        }


class _BuildTupleRankerNet(nn.Module):
    def __init__(self, input_dim: int, preview_dim: int, hidden_dim: int = 32):
        super().__init__()
        self.enc1 = nn.Linear(int(input_dim), int(hidden_dim))
        self.enc2 = nn.Linear(int(hidden_dim), int(hidden_dim))
        self.preview_enc = nn.Sequential(
            nn.Linear(int(hidden_dim) + int(preview_dim), int(hidden_dim)),
            nn.SiLU(),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.SiLU(),
        )
        self.preview_score = nn.Linear(int(hidden_dim), 1)
        self.state_value = nn.Linear(int(hidden_dim), 1)

    def forward(self, x: torch.Tensor, preview_x: torch.Tensor) -> dict[str, torch.Tensor]:
        state_hidden = F.silu(self.enc1(x))
        state_hidden = F.silu(self.enc2(state_hidden))
        state_expand = state_hidden.unsqueeze(1).expand(-1, preview_x.shape[1], -1)
        preview_hidden = self.preview_enc(torch.cat([state_expand, preview_x], dim=-1))
        return {
            "preview_score": self.preview_score(preview_hidden).squeeze(-1),
            "state_value": self.state_value(state_hidden).squeeze(-1),
        }


class _RepairControllerSharedNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        *,
        n_macro_actions: int,
        n_routes: int,
        path_input_dim: int,
        preview_input_dim: int = 0,
        provenance_input_dim: int = 0,
    ):
        super().__init__()
        hidden_dim = max(8, int(hidden_dim))
        self.enc1 = nn.Linear(int(input_dim), hidden_dim)
        self.enc2 = nn.Linear(hidden_dim, hidden_dim)
        self.aux_head = nn.Linear(hidden_dim, len(REPAIR_CRITIC_HEAD_NAMES))
        self.path_lin1 = nn.Linear(hidden_dim + int(path_input_dim), hidden_dim)
        self.path_lin2 = nn.Linear(hidden_dim, hidden_dim)
        self.policy_path_attn = nn.Linear(hidden_dim, 1)
        self.policy_fuse = nn.Linear(3 * hidden_dim, hidden_dim)
        self.policy_norm = nn.LayerNorm(hidden_dim)
        self.route_head = nn.Linear(hidden_dim, int(n_routes))
        self.macro_head = nn.Linear(hidden_dim, int(n_macro_actions))
        self.q_head = nn.Linear(hidden_dim, int(n_macro_actions))
        self.path_route_head = nn.Linear(hidden_dim, int(n_routes))
        self.path_macro_head = nn.Linear(hidden_dim, int(n_macro_actions))
        self.path_q_head = nn.Linear(hidden_dim, int(n_macro_actions))
        self.preview_input_dim = max(0, int(preview_input_dim))
        self.provenance_input_dim = max(0, int(provenance_input_dim))
        if self.preview_input_dim > 0:
            self.preview_lin1 = nn.Linear(hidden_dim + self.preview_input_dim, hidden_dim)
            self.preview_lin2 = nn.Linear(hidden_dim, hidden_dim)
            self.preview_utility_head = nn.Linear(hidden_dim, 1)
            self.preview_value_head = nn.Linear(hidden_dim, 1)
            self.preview_regret_head = nn.Linear(hidden_dim, 1)
        else:
            self.preview_lin1 = None
            self.preview_lin2 = None
            self.preview_utility_head = None
            self.preview_value_head = None
            self.preview_regret_head = None
        if self.preview_input_dim > 0 and self.provenance_input_dim > 0:
            self.provenance_lin1 = nn.Linear(hidden_dim + self.provenance_input_dim, hidden_dim)
            self.provenance_lin2 = nn.Linear(hidden_dim, hidden_dim)
            self.provenance_attn = nn.Linear(2 * hidden_dim, 1)
            self.provenance_fuse = nn.Linear(2 * hidden_dim, hidden_dim)
            self.provenance_norm = nn.LayerNorm(hidden_dim)
        else:
            self.provenance_lin1 = None
            self.provenance_lin2 = None
            self.provenance_attn = None
            self.provenance_fuse = None
            self.provenance_norm = None
        self.state_value_head = nn.Linear(hidden_dim, 1)
        self.path_out = nn.Linear(hidden_dim, 1)
        self.path_relation_head = nn.Linear(hidden_dim, len(REPAIR_CRITIC_PATH_RELATION_NAMES))
        self.path_mode_head = nn.Linear(hidden_dim, len(REPAIR_CRITIC_MODE_NAMES))
        self.path_improve_head = nn.Linear(hidden_dim, 1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.enc1(x))
        h = F.silu(self.enc2(h))
        return h

    def encode_path_candidates(self, shared: torch.Tensor, path_x: torch.Tensor) -> torch.Tensor:
        if path_x.ndim != 3:
            raise ValueError("path_x must have shape [batch, n_paths, path_dim].")
        shared_rep = shared.unsqueeze(1).expand(-1, path_x.shape[1], -1)
        z = torch.cat([shared_rep, path_x], dim=-1)
        z = F.silu(self.path_lin1(z))
        z = F.silu(self.path_lin2(z))
        return z

    def encode_preview_candidates(
        self,
        path_hidden: torch.Tensor,
        preview_x: torch.Tensor,
        preview_path_index: torch.Tensor,
    ) -> torch.Tensor:
        if self.preview_lin1 is None or self.preview_lin2 is None:
            raise ValueError("preview encoder is not configured for this model.")
        if preview_x.ndim != 3:
            raise ValueError("preview_x must have shape [batch, n_preview, preview_dim].")
        if preview_path_index.ndim != 2:
            raise ValueError("preview_path_index must have shape [batch, n_preview].")
        if path_hidden.ndim != 3:
            raise ValueError("path_hidden must have shape [batch, n_paths, hidden_dim].")
        preview_idx = preview_path_index.to(device=path_hidden.device, dtype=torch.long)
        preview_idx = preview_idx.clamp(min=0, max=max(0, path_hidden.shape[1] - 1))
        batch_idx = torch.arange(path_hidden.shape[0], device=path_hidden.device).unsqueeze(1).expand_as(preview_idx)
        gathered_path_hidden = path_hidden[batch_idx, preview_idx]
        z = torch.cat([gathered_path_hidden, preview_x], dim=-1)
        z = F.silu(self.preview_lin1(z))
        z = F.silu(self.preview_lin2(z))
        return z

    def encode_provenance_candidates(
        self,
        path_hidden: torch.Tensor,
        provenance_x: torch.Tensor,
        provenance_path_index: torch.Tensor,
    ) -> torch.Tensor:
        if self.provenance_lin1 is None or self.provenance_lin2 is None:
            raise ValueError("provenance encoder is not configured for this model.")
        if provenance_x.ndim != 4:
            raise ValueError("provenance_x must have shape [batch, n_preview, n_prov, provenance_dim].")
        if provenance_path_index.ndim != 2:
            raise ValueError("provenance_path_index must have shape [batch, n_preview].")
        if path_hidden.ndim != 3:
            raise ValueError("path_hidden must have shape [batch, n_paths, hidden_dim].")
        prov_idx = provenance_path_index.to(device=path_hidden.device, dtype=torch.long)
        prov_idx = prov_idx.clamp(min=0, max=max(0, path_hidden.shape[1] - 1))
        batch_idx = torch.arange(path_hidden.shape[0], device=path_hidden.device).unsqueeze(1).expand_as(prov_idx)
        gathered_path_hidden = path_hidden[batch_idx, prov_idx]
        gathered_path_hidden = gathered_path_hidden.unsqueeze(2).expand(-1, -1, provenance_x.shape[2], -1)
        z = torch.cat([gathered_path_hidden, provenance_x], dim=-1)
        z = F.silu(self.provenance_lin1(z))
        z = F.silu(self.provenance_lin2(z))
        return z

    def _fuse_provenance_context(
        self,
        preview_hidden: torch.Tensor,
        provenance_hidden: torch.Tensor,
        provenance_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.provenance_attn is None or self.provenance_fuse is None or self.provenance_norm is None:
            return preview_hidden, None
        if provenance_mask is None:
            provenance_mask = torch.ones(
                provenance_hidden.shape[:3],
                dtype=torch.bool,
                device=provenance_hidden.device,
            )
        else:
            provenance_mask = provenance_mask.to(device=provenance_hidden.device, dtype=torch.bool)
        preview_rep = preview_hidden.unsqueeze(2).expand(-1, -1, provenance_hidden.shape[2], -1)
        attn_logits = self.provenance_attn(torch.cat([preview_rep, provenance_hidden], dim=-1)).squeeze(-1)
        masked_logits = attn_logits.masked_fill(~provenance_mask, -1.0e9)
        attn = torch.softmax(masked_logits, dim=-1)
        attn = torch.where(provenance_mask, attn, torch.zeros_like(attn))
        denom = attn.sum(dim=-1, keepdim=True)
        attn = torch.where(denom > 0.0, attn / denom.clamp_min(1.0e-12), attn)
        pooled = torch.sum(attn.unsqueeze(-1) * provenance_hidden, dim=2)
        fused = F.silu(self.provenance_fuse(torch.cat([preview_hidden, pooled], dim=-1)))
        preview_hidden_fused = self.provenance_norm(preview_hidden + fused)
        has_provenance = provenance_mask.any(dim=-1, keepdim=True)
        preview_hidden_fused = torch.where(has_provenance, preview_hidden_fused, preview_hidden)
        return preview_hidden_fused, attn

    def _policy_context(
        self,
        shared: torch.Tensor,
        path_hidden: torch.Tensor | None,
        path_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if path_hidden is None:
            return shared, None
        if path_mask is None:
            path_mask = torch.ones(
                (path_hidden.shape[0], path_hidden.shape[1]),
                dtype=torch.bool,
                device=path_hidden.device,
            )
        else:
            path_mask = path_mask.to(device=path_hidden.device, dtype=torch.bool)
        attn_logits = self.policy_path_attn(path_hidden).squeeze(-1)
        attn = _masked_path_probs(attn_logits, path_mask)
        pooled = torch.sum(attn.unsqueeze(-1) * path_hidden, dim=1)
        masked_hidden = path_hidden.masked_fill(~path_mask.unsqueeze(-1), float("-inf"))
        top_hidden = masked_hidden.max(dim=1).values
        top_hidden = torch.where(torch.isfinite(top_hidden), top_hidden, torch.zeros_like(top_hidden))
        fused = torch.cat([shared, pooled, top_hidden], dim=-1)
        policy_hidden = F.silu(self.policy_fuse(fused))
        policy_hidden = self.policy_norm(policy_hidden + shared)
        has_path = path_mask.any(dim=1, keepdim=True)
        policy_hidden = torch.where(has_path, policy_hidden, shared)
        return policy_hidden, attn

    def forward(
        self,
        x: torch.Tensor,
        path_x: torch.Tensor | None = None,
        path_mask: torch.Tensor | None = None,
        preview_x: torch.Tensor | None = None,
        preview_path_index: torch.Tensor | None = None,
        preview_mask: torch.Tensor | None = None,
        provenance_x: torch.Tensor | None = None,
        provenance_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        shared = self.encode(x)
        decision_hidden = shared
        out = {
            "shared": shared,
            "aux_logits": self.aux_head(shared),
        }
        if path_x is not None:
            path_hidden = self.encode_path_candidates(shared, path_x)
            out["path_hidden"] = path_hidden
            decision_hidden, path_policy_weights = self._policy_context(shared, path_hidden, path_mask)
            out["path_policy_weights"] = path_policy_weights
            out["path_logits"] = self.path_out(path_hidden).squeeze(-1)
            out["path_route_logits"] = self.path_route_head(path_hidden)
            out["path_macro_logits"] = self.path_macro_head(path_hidden)
            out["path_q_values"] = self.path_q_head(path_hidden)
            out["path_relation_logits"] = self.path_relation_head(path_hidden)
            out["path_mode_logits"] = self.path_mode_head(path_hidden)
            out["path_improve"] = self.path_improve_head(path_hidden).squeeze(-1)
            if (
                preview_x is not None
                and preview_path_index is not None
                and self.preview_lin1 is not None
                and self.preview_utility_head is not None
            ):
                preview_hidden = self.encode_preview_candidates(path_hidden, preview_x, preview_path_index)
                if (
                    provenance_x is not None
                    and self.provenance_lin1 is not None
                    and preview_path_index is not None
                ):
                    provenance_hidden = self.encode_provenance_candidates(path_hidden, provenance_x, preview_path_index)
                    preview_hidden, provenance_weights = self._fuse_provenance_context(
                        preview_hidden,
                        provenance_hidden,
                        provenance_mask,
                    )
                    out["provenance_hidden"] = provenance_hidden
                    out["provenance_weights"] = provenance_weights
                out["preview_hidden"] = preview_hidden
                out["preview_utility"] = self.preview_utility_head(preview_hidden).squeeze(-1)
                out["preview_value"] = self.preview_value_head(preview_hidden).squeeze(-1)
                out["preview_regret"] = F.softplus(self.preview_regret_head(preview_hidden).squeeze(-1))
                if preview_mask is not None:
                    out["preview_mask"] = preview_mask.to(device=preview_hidden.device, dtype=torch.bool)
        out["policy_shared"] = decision_hidden
        out["route_logits"] = self.route_head(decision_hidden)
        out["macro_logits"] = self.macro_head(decision_hidden)
        out["q_values"] = self.q_head(decision_hidden)
        out["value_pred"] = self.state_value_head(decision_hidden).squeeze(-1)
        return out


class _SharedCandidateDualRankerNet(_RepairControllerSharedNet):
    """Shared candidate encoder with separate repair/build ranking heads."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        *,
        path_input_dim: int,
        preview_input_dim: int,
        provenance_input_dim: int = 0,
    ):
        super().__init__(
            input_dim,
            hidden_dim,
            n_macro_actions=1,
            n_routes=1,
            path_input_dim=path_input_dim,
            preview_input_dim=preview_input_dim,
            provenance_input_dim=provenance_input_dim,
        )
        self.repair_preview_score_head = nn.Linear(hidden_dim, 1)
        self.build_preview_score_head = nn.Linear(hidden_dim, 1)
        self.common_preview_q_head = nn.Linear(hidden_dim, 1)
        self.repair_state_value_head = nn.Linear(hidden_dim, 1)
        self.build_state_value_head = nn.Linear(hidden_dim, 1)
        self.oracle_preview_truth_head = nn.Linear(hidden_dim, 1)
        self.oracle_preview_mode_best_head = nn.Linear(hidden_dim, 1)
        self.oracle_preview_rank_head = nn.Linear(hidden_dim, 1)
        self.oracle_preview_stability_head = nn.Linear(hidden_dim, 1)
        self.oracle_state_coverage_head = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        x: torch.Tensor,
        path_x: torch.Tensor | None = None,
        path_mask: torch.Tensor | None = None,
        preview_x: torch.Tensor | None = None,
        preview_path_index: torch.Tensor | None = None,
        preview_mask: torch.Tensor | None = None,
        provenance_x: torch.Tensor | None = None,
        provenance_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        out = super().forward(
            x,
            path_x,
            path_mask,
            preview_x,
            preview_path_index,
            preview_mask,
            provenance_x,
            provenance_mask,
        )
        preview_hidden = out.get("preview_hidden", None)
        if preview_hidden is not None:
            out["repair_preview_score"] = self.repair_preview_score_head(preview_hidden).squeeze(-1)
            out["build_preview_score"] = self.build_preview_score_head(preview_hidden).squeeze(-1)
            out["common_preview_q"] = self.common_preview_q_head(preview_hidden).squeeze(-1)
            out["oracle_preview_truth_logit"] = self.oracle_preview_truth_head(preview_hidden).squeeze(-1)
            out["oracle_preview_mode_best_logit"] = self.oracle_preview_mode_best_head(preview_hidden).squeeze(-1)
            out["oracle_preview_rank_logit"] = self.oracle_preview_rank_head(preview_hidden).squeeze(-1)
            out["oracle_preview_stability_logit"] = self.oracle_preview_stability_head(preview_hidden).squeeze(-1)
        policy_shared = out.get("policy_shared", out["shared"])
        out["repair_state_value"] = self.repair_state_value_head(policy_shared).squeeze(-1)
        out["build_state_value"] = self.build_state_value_head(policy_shared).squeeze(-1)
        out["oracle_state_coverage_logit"] = self.oracle_state_coverage_head(policy_shared).squeeze(-1)
        return out


def _binary_head_pos_weight(y: torch.Tensor) -> torch.Tensor:
    pos = float((y > 0.5).sum().item())
    neg = float(y.numel() - pos)
    if pos <= 0.0 or neg <= 0.0:
        return torch.tensor(1.0, dtype=y.dtype, device=y.device)
    return torch.tensor(max(1.0, neg / max(pos, 1.0)), dtype=y.dtype, device=y.device)


def _split_indices(n: int, val_fraction: float, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    idx = torch.arange(n, dtype=torch.long)
    if n <= 1 or val_fraction <= 0.0:
        return idx, idx[:0]
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))
    perm = idx[torch.randperm(n, generator=g)]
    n_val = max(1, int(round(float(n) * float(val_fraction))))
    n_val = min(max(1, n_val), max(1, n - 1))
    return perm[n_val:], perm[:n_val]


def _aux_predictions_from_logits(
    logits: torch.Tensor,
    *,
    head_names: Sequence[str] = REPAIR_CRITIC_HEAD_NAMES,
) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for i, name in enumerate(head_names):
        out[str(name)] = torch.sigmoid(logits[..., i])
    return out


def _metrics_from_preds(
    preds: dict[str, torch.Tensor],
    y: torch.Tensor,
    *,
    head_names: Sequence[str] = REPAIR_CRITIC_HEAD_NAMES,
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for i, name in enumerate(head_names):
        target = y[:, i]
        pred = preds[str(name)]
        mae = torch.mean(torch.abs(pred - target)).item()
        metrics[f"{name}_mae"] = float(mae)
        if name not in ("reward_per_s_score", "utility_score"):
            acc = torch.mean(((pred >= 0.5) == (target >= 0.5)).to(torch.float32)).item()
            metrics[f"{name}_acc"] = float(acc)
    return metrics


def _aux_loss_from_logits(
    logits: torch.Tensor,
    y: torch.Tensor,
    *,
    pos_weights: Sequence[torch.Tensor],
) -> torch.Tensor:
    if logits.shape[0] == 0:
        return logits.sum() * 0.0
    loss = 0.0
    for i in range(4):
        loss = loss + F.binary_cross_entropy_with_logits(
            logits[:, i],
            y[:, i],
            pos_weight=pos_weights[i],
        )
    reward_pred = torch.sigmoid(logits[:, 4])
    utility_pred = torch.sigmoid(logits[:, 5])
    loss = loss + F.mse_loss(reward_pred, y[:, 4])
    loss = loss + 1.5 * F.mse_loss(utility_pred, y[:, 5])
    return loss


def _macro_metrics_from_logits(
    logits: torch.Tensor,
    y: torch.Tensor,
    *,
    action_names: Sequence[str],
) -> dict[str, float]:
    if logits.shape[0] == 0:
        return {}
    probs = torch.softmax(logits, dim=-1)
    pred_idx = probs.argmax(dim=-1)
    label_prob = probs.gather(1, y.reshape(-1, 1)).reshape(-1)
    return {
        "macro_action_acc": float(torch.mean((pred_idx == y).to(torch.float32)).item()),
        "macro_action_label_prob": float(torch.mean(label_prob).item()),
        "macro_action_classes": float(len(tuple(action_names))),
    }


def _inverse_frequency_class_weights(y: torch.Tensor, n_classes: int) -> torch.Tensor:
    n_classes = max(1, int(n_classes))
    weights = torch.ones((n_classes,), dtype=torch.float32, device=y.device)
    if y.numel() <= 0:
        return weights
    counts = torch.bincount(y.reshape(-1).to(torch.long), minlength=n_classes).to(torch.float32)
    valid = counts > 0.0
    if bool(valid.any().item()):
        total = float(counts[valid].sum().item())
        denom = float(valid.sum().item())
        weights = torch.where(valid, total / (counts.clamp_min(1.0) * max(1.0, denom)), weights)
        weights = weights / weights.mean().clamp_min(1.0e-6)
    return weights


def _route_action_mask(
    action_names: Sequence[str],
    route_names: Sequence[str],
) -> torch.Tensor:
    route_index = {str(name): idx for idx, name in enumerate(route_names)}
    mask = torch.zeros((len(tuple(route_names)), len(tuple(action_names))), dtype=torch.bool)
    for action_idx, action_name in enumerate(action_names):
        route_name = REPAIR_CRITIC_ACTION_ROUTE_MAP.get(str(action_name), None)
        if route_name in route_index:
            mask[int(route_index[route_name]), int(action_idx)] = True
    return mask


def _route_masked_action_logits(
    logits: torch.Tensor,
    route_targets: torch.Tensor,
    route_action_mask: torch.Tensor,
) -> torch.Tensor:
    if logits.shape[0] == 0:
        return logits
    mask = route_action_mask[route_targets.to(torch.long)]
    return logits.masked_fill(~mask, -1.0e9)


def _hierarchical_macro_metrics(
    route_logits: torch.Tensor,
    macro_logits: torch.Tensor,
    y_route: torch.Tensor,
    y_action: torch.Tensor,
    route_action_mask: torch.Tensor,
) -> dict[str, float]:
    if route_logits.shape[0] == 0 or macro_logits.shape[0] == 0:
        return {}
    masked_true_logits = _route_masked_action_logits(macro_logits, y_route, route_action_mask)
    true_route_probs = torch.softmax(masked_true_logits, dim=-1)
    true_route_pred = true_route_probs.argmax(dim=-1)
    true_route_label_prob = true_route_probs.gather(1, y_action.reshape(-1, 1)).reshape(-1)

    pred_route = torch.softmax(route_logits, dim=-1).argmax(dim=-1)
    masked_pred_logits = _route_masked_action_logits(macro_logits, pred_route, route_action_mask)
    pred_action = torch.softmax(masked_pred_logits, dim=-1).argmax(dim=-1)
    return {
        "macro_action_within_route_acc": float(torch.mean((true_route_pred == y_action).to(torch.float32)).item()),
        "macro_action_within_route_label_prob": float(torch.mean(true_route_label_prob).item()),
        "macro_action_hier_acc": float(torch.mean((pred_action == y_action).to(torch.float32)).item()),
    }


def _masked_path_cross_entropy(
    logits: torch.Tensor,
    y: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    if logits.shape[0] == 0:
        return logits.sum() * 0.0
    masked_logits = logits.masked_fill(~valid_mask, -1.0e9)
    return F.cross_entropy(masked_logits, y)


def _masked_path_probs(logits: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    masked_logits = logits.masked_fill(~valid_mask, -1.0e9)
    probs = torch.softmax(masked_logits, dim=-1)
    probs = torch.where(valid_mask, probs, torch.zeros_like(probs))
    denom = probs.sum(dim=-1, keepdim=True)
    return torch.where(denom > 0.0, probs / denom.clamp_min(1.0e-12), probs)


def _gather_path_head(path_tensor: torch.Tensor, y_path: torch.Tensor) -> torch.Tensor:
    if path_tensor.ndim < 3:
        raise ValueError("path_tensor must have shape [batch, n_paths, ...].")
    index = y_path.to(torch.long).reshape(-1, 1, *([1] * (path_tensor.ndim - 2)))
    index = index.expand(-1, 1, *path_tensor.shape[2:])
    return path_tensor.gather(1, index).squeeze(1)


def _path_metrics_from_logits(logits: torch.Tensor, y: torch.Tensor, valid_mask: torch.Tensor) -> dict[str, float]:
    if logits.shape[0] == 0:
        return {}
    probs = _masked_path_probs(logits, valid_mask)
    pred_idx = probs.argmax(dim=-1)
    label_prob = probs.gather(1, y.reshape(-1, 1)).reshape(-1)
    return {
        "path_head_acc": float(torch.mean((pred_idx == y).to(torch.float32)).item()),
        "path_head_label_prob": float(torch.mean(label_prob).item()),
        "path_head_mean_choices": float(torch.mean(valid_mask.to(torch.float32).sum(dim=-1)).item()),
    }


def _flat_classification_metrics(
    logits: torch.Tensor,
    y: torch.Tensor,
    *,
    prefix: str,
) -> dict[str, float]:
    if logits.shape[0] == 0:
        return {}
    probs = torch.softmax(logits, dim=-1)
    pred_idx = probs.argmax(dim=-1)
    label_prob = probs.gather(1, y.reshape(-1, 1)).reshape(-1)
    return {
        f"{prefix}_acc": float(torch.mean((pred_idx == y).to(torch.float32)).item()),
        f"{prefix}_label_prob": float(torch.mean(label_prob).item()),
    }


def _binary_classification_metrics(
    logits: torch.Tensor,
    y: torch.Tensor,
    *,
    prefix: str,
) -> dict[str, float]:
    if logits.numel() == 0 or y.numel() == 0:
        return {}
    probs = torch.sigmoid(logits.reshape(-1))
    preds = (probs >= 0.5).to(dtype=torch.long)
    target = y.reshape(-1).to(dtype=torch.long)
    acc = float((preds == target).float().mean().item())
    pos_rate = float(target.float().mean().item())
    pred_rate = float(preds.float().mean().item())
    return {
        f"{prefix}_acc": acc,
        f"{prefix}_target_pos_rate": pos_rate,
        f"{prefix}_pred_pos_rate": pred_rate,
    }


def _regression_metrics(pred: torch.Tensor, y: torch.Tensor, *, prefix: str) -> dict[str, float]:
    if pred.shape[0] == 0:
        return {}
    mae = torch.mean(torch.abs(pred - y)).item()
    return {f"{prefix}_mae": float(mae)}


def _gather_path_action_scores(
    path_q_values: torch.Tensor,
    action_indices: torch.Tensor,
) -> torch.Tensor:
    if path_q_values.ndim != 3:
        raise ValueError("path_q_values must have shape [batch, n_paths, n_actions].")
    if action_indices.ndim != 2:
        raise ValueError("action_indices must have shape [batch, n_paths].")
    index = action_indices.to(torch.long).unsqueeze(-1)
    return path_q_values.gather(-1, index).squeeze(-1)


def _masked_log_softmax(logits: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    masked_logits = logits.masked_fill(~valid_mask, -1.0e9)
    log_probs = torch.log_softmax(masked_logits, dim=-1)
    return torch.where(valid_mask, log_probs, torch.zeros_like(log_probs))


def _listwise_slate_loss(
    scores: torch.Tensor,
    target_scores: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    beta: float,
    min_gap: float,
) -> torch.Tensor:
    if scores.shape[0] == 0:
        return scores.sum() * 0.0
    row_span = (
        target_scores.masked_fill(~valid_mask, float("-inf")).max(dim=-1).values
        - target_scores.masked_fill(~valid_mask, float("inf")).min(dim=-1).values
    )
    informative = valid_mask.sum(dim=-1) >= 2
    informative = informative & torch.isfinite(row_span) & (row_span > float(min_gap))
    if not bool(informative.any().item()):
        return scores.sum() * 0.0
    scores = scores[informative]
    target_scores = target_scores[informative]
    valid_mask = valid_mask[informative]
    pred_log_probs = _masked_log_softmax(scores, valid_mask)
    target_logits = float(beta) * target_scores
    target_probs = _masked_path_probs(target_logits, valid_mask)
    return -(target_probs * pred_log_probs).sum(dim=-1).mean()


def _pairwise_rank_loss(
    scores: torch.Tensor,
    pairwise_pairs: Sequence[Sequence[tuple[int, int, float]]],
    *,
    gap_scale: float,
) -> torch.Tensor:
    if scores.shape[0] == 0:
        return scores.sum() * 0.0
    losses: list[torch.Tensor] = []
    weights: list[float] = []
    scale = max(1.0e-6, float(gap_scale))
    for row_idx, pairs in enumerate(pairwise_pairs):
        for better_idx, worse_idx, gap in list(pairs or ()):
            try:
                better = int(better_idx)
                worse = int(worse_idx)
                gap_f = float(gap)
            except Exception:
                continue
            if better < 0 or worse < 0 or better >= scores.shape[1] or worse >= scores.shape[1]:
                continue
            diff = scores[row_idx, better] - scores[row_idx, worse]
            losses.append(F.softplus(-diff))
            weights.append(float(min(4.0, max(0.25, gap_f / scale))))
    if not losses:
        return scores.sum() * 0.0
    weight_t = scores.new_tensor(weights)
    loss_t = torch.stack(losses)
    return (weight_t * loss_t).sum() / weight_t.sum().clamp_min(1.0e-6)


def _masked_pairwise_pairs_from_targets(
    target_scores: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    gap_floor: float = 1.0e-3,
) -> list[list[tuple[int, int, float]]]:
    if target_scores.ndim != 2 or valid_mask.ndim != 2:
        raise ValueError("target_scores and valid_mask must have shape [batch, n_items].")
    score_cpu = target_scores.detach().cpu()
    mask_cpu = valid_mask.detach().cpu().to(torch.bool)
    out: list[list[tuple[int, int, float]]] = []
    for row_idx in range(score_cpu.shape[0]):
        row_pairs: list[tuple[int, int, float]] = []
        valid_idx = [int(idx) for idx, keep in enumerate(mask_cpu[row_idx].tolist()) if bool(keep)]
        for ii, left_idx in enumerate(valid_idx):
            left_val = float(score_cpu[row_idx, left_idx].item())
            for right_idx in valid_idx[ii + 1 :]:
                right_val = float(score_cpu[row_idx, right_idx].item())
                gap = float(left_val - right_val)
                if abs(gap) <= float(gap_floor):
                    continue
                if gap > 0.0:
                    row_pairs.append((int(left_idx), int(right_idx), float(gap)))
                else:
                    row_pairs.append((int(right_idx), int(left_idx), float(-gap)))
        out.append(row_pairs)
    return out


def _slate_rank_metrics(
    scores: torch.Tensor,
    target_scores: torch.Tensor,
    valid_mask: torch.Tensor,
    pairwise_pairs: Sequence[Sequence[tuple[int, int, float]]],
    *,
    prefix: str = "slate",
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    if scores.shape[0] == 0:
        return metrics
    valid_mask = valid_mask.to(torch.bool)
    if bool(valid_mask.any().item()):
        pred_idx = scores.masked_fill(~valid_mask, float("-inf")).argmax(dim=-1)
        target_idx = target_scores.masked_fill(~valid_mask, float("-inf")).argmax(dim=-1)
        informative = valid_mask.sum(dim=-1) >= 1
        if bool(informative.any().item()):
            metrics[f"{prefix}_top1_acc"] = float(
                torch.mean((pred_idx[informative] == target_idx[informative]).to(torch.float32)).item()
            )
    total_pairs = 0
    correct_pairs = 0
    for row_idx, pairs in enumerate(pairwise_pairs):
        for better_idx, worse_idx, _gap in list(pairs or ()):
            try:
                better = int(better_idx)
                worse = int(worse_idx)
            except Exception:
                continue
            if better < 0 or worse < 0 or better >= scores.shape[1] or worse >= scores.shape[1]:
                continue
            total_pairs += 1
            if float(scores[row_idx, better].item()) > float(scores[row_idx, worse].item()):
                correct_pairs += 1
    if total_pairs > 0:
        metrics[f"{prefix}_pairwise_acc"] = float(correct_pairs / float(total_pairs))
        metrics[f"{prefix}_pairwise_pairs"] = float(total_pairs)
    return metrics


def _route_emergence_metrics(
    scores: torch.Tensor,
    target_scores: torch.Tensor,
    valid_mask: torch.Tensor,
    route_targets: Sequence[Sequence[str]],
    *,
    prefix: str,
    tau: float = 1.0,
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    if scores.shape[0] == 0:
        return metrics
    top1_hits = 0
    lse_hits = 0
    n_rows = 0
    regret_sum = 0.0
    tau_f = max(1.0e-6, float(tau))
    for row_idx in range(scores.shape[0]):
        valid_idx = [idx for idx, keep in enumerate(valid_mask[row_idx].tolist()) if bool(keep)]
        if len(valid_idx) < 2:
            continue
        routes = list(route_targets[row_idx]) if row_idx < len(route_targets) else []
        if len(routes) < scores.shape[1]:
            routes = routes + [""] * (scores.shape[1] - len(routes))
        best_exact_idx = max(valid_idx, key=lambda idx: float(target_scores[row_idx, idx].item()))
        best_exact_route = str(routes[best_exact_idx] or "")
        pred_top1_idx = max(valid_idx, key=lambda idx: float(scores[row_idx, idx].item()))
        pred_top1_route = str(routes[pred_top1_idx] or "")
        route_groups: dict[str, list[int]] = {}
        for idx in valid_idx:
            route_name = str(routes[idx] or "")
            route_groups.setdefault(route_name, []).append(idx)
        if len(route_groups) < 2:
            continue
        route_lse: dict[str, float] = {}
        route_best_target: dict[str, float] = {}
        for route_name, idxs in route_groups.items():
            route_score_tensor = torch.stack([scores[row_idx, idx] / tau_f for idx in idxs])
            route_lse[route_name] = float(torch.logsumexp(route_score_tensor, dim=0).item())
            route_best_target[route_name] = max(float(target_scores[row_idx, idx].item()) for idx in idxs)
        pred_lse_route = max(route_lse, key=lambda name: (route_lse[name], name))
        exact_best_route_utility = max(route_best_target.values())
        chosen_route_utility = float(route_best_target.get(pred_lse_route, exact_best_route_utility))
        regret_sum += max(0.0, exact_best_route_utility - chosen_route_utility)
        top1_hits += int(pred_top1_route == best_exact_route)
        lse_hits += int(pred_lse_route == best_exact_route)
        n_rows += 1
    if n_rows > 0:
        metrics[f"{prefix}_top1_route_acc"] = float(top1_hits / float(n_rows))
        metrics[f"{prefix}_lse_route_acc"] = float(lse_hits / float(n_rows))
        metrics[f"{prefix}_mean_route_regret"] = float(regret_sum / float(n_rows))
        metrics[f"{prefix}_route_examples"] = float(n_rows)
    return metrics


def _compute_feature_stats(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if x.shape[0] == 0:
        return torch.zeros((x.shape[1],), dtype=x.dtype), torch.ones((x.shape[1],), dtype=x.dtype)
    mean = x.mean(dim=0)
    std = x.std(dim=0, unbiased=False)
    std = torch.where(std > 1.0e-6, std, torch.ones_like(std))
    return mean, std


def _normalize_inputs(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    std = torch.where(std > 1.0e-6, std, torch.ones_like(std))
    return (x - mean) / std


def _maybe_init_model_from_bundle(model: nn.Module, bundle: dict[str, Any] | None) -> None:
    if not isinstance(bundle, dict):
        return
    state_dict = bundle.get("model_state_dict", None)
    if not isinstance(state_dict, dict):
        return
    patched_state = dict(state_dict)
    if ("value_head.weight" in patched_state) and ("state_value_head.weight" not in patched_state):
        patched_state["state_value_head.weight"] = patched_state["value_head.weight"]
        patched_state["preview_value_head.weight"] = patched_state["value_head.weight"]
        if "value_head.bias" in patched_state:
            patched_state["state_value_head.bias"] = patched_state["value_head.bias"]
            patched_state["preview_value_head.bias"] = patched_state["value_head.bias"]
    model_state = model.state_dict()
    compatible_state = {
        key: value
        for key, value in patched_state.items()
        if key in model_state and tuple(model_state[key].shape) == tuple(value.shape)
    }
    model.load_state_dict(compatible_state, strict=False)


def _bundle_kind_from_payload(payload: dict[str, Any]) -> str:
    model_kind = str(payload.get("model_kind", "") or "")
    if model_kind:
        return model_kind
    if bool(payload.get("shared_candidate_dual_trained", False)):
        return REPAIR_CRITIC_SHARED_CANDIDATE_MODEL_KIND
    if bool(payload.get("unified_candidate_ranker_trained", False)):
        return REPAIR_CRITIC_UNIFIED_CANDIDATE_MODEL_KIND
    if bool(payload.get("build_tuple_ranker_trained", False)):
        return REPAIR_CRITIC_BUILD_TUPLE_MODEL_KIND
    if bool(payload.get("repair_build_route_compare_trained", False)):
        return REPAIR_CRITIC_ROUTE_COMPARE_MODEL_KIND
    model_state = payload.get("model_state_dict", {})
    if any(str(key).startswith("aux_head.") or str(key).startswith("enc1.") for key in model_state):
        return REPAIR_CRITIC_SHARED_MODEL_KIND
    return REPAIR_CRITIC_LEGACY_MODEL_KIND
