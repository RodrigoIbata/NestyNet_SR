# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from .oracle_lab import (
    default_oracle_hyperparams,
    generate_oracle_shared_candidate_pretrain_dataset,
)
from .oracle_suite import _resolve_spec_paths
from .repair_critic import (
    load_inverse_experiment_rows,
    load_repair_critic_bundle,
    predict_shared_candidate_dual_slate,
    save_repair_critic_bundle,
    train_shared_candidate_dual_ranker,
)


def _write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_shared_candidate_oracle_pretrain_pipeline(
    *,
    spec_glob: str,
    output_dir: str,
    seeds: list[int],
    depth_min: int = 3,
    depth_max: int = 8,
    topk: int = 8,
    max_corrupt_paths_per_spec: int | None = None,
    sweep_max_paths: int | None = None,
    hidden_dim: int = 32,
    pretrain_epochs: int = 120,
    pretrain_lr: float = 5.0e-3,
    pretrain_seed: int = 0,
    pretrain_repair_rank_weight: float = 0.15,
    pretrain_build_rank_weight: float = 0.15,
    pretrain_common_rank_weight: float = 0.35,
    pretrain_path_ce_weight: float = 0.02,
    pretrain_common_value_weight: float = 0.15,
    pretrain_preview_value_weight: float = 0.02,
    pretrain_preview_regret_weight: float = 0.05,
    pretrain_common_state_value_weight: float = 0.02,
    pretrain_repair_state_value_weight: float = 0.0,
    pretrain_build_state_value_weight: float = 0.0,
    pretrain_oracle_path_weight: float = 0.15,
    pretrain_oracle_relation_weight: float = 0.10,
    pretrain_oracle_mode_weight: float = 0.10,
    pretrain_oracle_truth_weight: float = 0.10,
    pretrain_oracle_mode_best_weight: float = 0.10,
    pretrain_oracle_rank_weight: float = 0.15,
    pretrain_oracle_stability_weight: float = 0.12,
    pretrain_oracle_coverage_weight: float = 0.10,
    finetune_report_paths: list[str] | None = None,
    finetune_epochs: int = 120,
    finetune_lr: float = 5.0e-3,
    finetune_seed: int = 0,
    quiet: bool = False,
) -> dict[str, Any]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    hp = default_oracle_hyperparams()
    hp.n_fit = 32
    hp.n_probe = 48
    hp.inverse_max_paths = max(4, int(hp.inverse_max_paths))
    hp.inverse_topk_terms = max(3, int(hp.inverse_topk_terms))
    hp.inverse_shortlist_mult = max(2, int(hp.inverse_shortlist_mult))

    spec_paths = _resolve_spec_paths(None, spec_glob)
    payload = generate_oracle_shared_candidate_pretrain_dataset(
        spec_paths,
        factorized_search_hp=hp,
        seeds=tuple(int(s) for s in seeds),
        dtype=torch.float64,
        depth_min=int(depth_min),
        depth_max=int(depth_max),
        topk=int(topk),
        max_corrupt_paths_per_spec=max_corrupt_paths_per_spec,
        sweep_max_paths=sweep_max_paths,
        verbose=not bool(quiet),
    )
    dataset_path = out_dir / "oracle_shared_candidate_pretrain_dataset.json"
    _write_json(payload, dataset_path)

    pretrain_bundle = train_shared_candidate_dual_ranker(
        payload["rows"],
        hidden_dim=int(hidden_dim),
        epochs=int(pretrain_epochs),
        lr=float(pretrain_lr),
        seed=int(pretrain_seed),
        repair_rank_weight=float(pretrain_repair_rank_weight),
        build_rank_weight=float(pretrain_build_rank_weight),
        common_rank_weight=float(pretrain_common_rank_weight),
        path_ce_weight=float(pretrain_path_ce_weight),
        common_value_weight=float(pretrain_common_value_weight),
        preview_value_weight=float(pretrain_preview_value_weight),
        preview_regret_weight=float(pretrain_preview_regret_weight),
        common_state_value_weight=float(pretrain_common_state_value_weight),
        repair_state_value_weight=float(pretrain_repair_state_value_weight),
        build_state_value_weight=float(pretrain_build_state_value_weight),
        oracle_path_weight=float(pretrain_oracle_path_weight),
        oracle_relation_weight=float(pretrain_oracle_relation_weight),
        oracle_mode_weight=float(pretrain_oracle_mode_weight),
        oracle_truth_weight=float(pretrain_oracle_truth_weight),
        oracle_mode_best_weight=float(pretrain_oracle_mode_best_weight),
        oracle_rank_weight=float(pretrain_oracle_rank_weight),
        oracle_stability_weight=float(pretrain_oracle_stability_weight),
        oracle_coverage_weight=float(pretrain_oracle_coverage_weight),
    )
    pretrain_bundle_path = out_dir / "oracle_shared_candidate_pretrain_bundle.pt"
    save_repair_critic_bundle(pretrain_bundle, pretrain_bundle_path)

    final_bundle = pretrain_bundle
    final_bundle_path = pretrain_bundle_path
    finetune_summary: dict[str, Any] | None = None
    if finetune_report_paths:
        rows = load_inverse_experiment_rows(finetune_report_paths)
        final_bundle = train_shared_candidate_dual_ranker(
            rows,
            hidden_dim=int(hidden_dim),
            epochs=int(finetune_epochs),
            lr=float(finetune_lr),
            seed=int(finetune_seed),
            init_bundle=pretrain_bundle,
        )
        final_bundle_path = out_dir / "oracle_shared_candidate_finetuned_bundle.pt"
        save_repair_critic_bundle(final_bundle, final_bundle_path)
        pred_bundle = load_repair_critic_bundle(final_bundle_path)
        sample_pred = None
        for row in rows:
            pred = predict_shared_candidate_dual_slate(pred_bundle, row)
            if pred.get("trained", False):
                sample_pred = pred
                break
        finetune_summary = {
            "n_rows": int(len(rows)),
            "sample_prediction": sample_pred,
            "metrics": dict(final_bundle.get("metrics", {}) or {}),
        }
        _write_json(finetune_summary, out_dir / "oracle_shared_candidate_finetune_summary.json")

    summary = {
        "spec_glob": str(spec_glob),
        "n_specs": int(len(spec_paths)),
        "n_curriculum_rows": int(payload.get("n_rows", 0)),
        "dataset_path": str(dataset_path),
        "pretrain_bundle_path": str(pretrain_bundle_path),
        "final_bundle_path": str(final_bundle_path),
        "pretrain_loss_config": {
            "repair_rank_weight": float(pretrain_repair_rank_weight),
            "build_rank_weight": float(pretrain_build_rank_weight),
            "common_rank_weight": float(pretrain_common_rank_weight),
            "path_ce_weight": float(pretrain_path_ce_weight),
            "common_value_weight": float(pretrain_common_value_weight),
            "preview_value_weight": float(pretrain_preview_value_weight),
            "preview_regret_weight": float(pretrain_preview_regret_weight),
            "common_state_value_weight": float(pretrain_common_state_value_weight),
            "repair_state_value_weight": float(pretrain_repair_state_value_weight),
            "build_state_value_weight": float(pretrain_build_state_value_weight),
            "oracle_path_weight": float(pretrain_oracle_path_weight),
            "oracle_relation_weight": float(pretrain_oracle_relation_weight),
            "oracle_mode_weight": float(pretrain_oracle_mode_weight),
            "oracle_truth_weight": float(pretrain_oracle_truth_weight),
            "oracle_mode_best_weight": float(pretrain_oracle_mode_best_weight),
            "oracle_rank_weight": float(pretrain_oracle_rank_weight),
            "oracle_stability_weight": float(pretrain_oracle_stability_weight),
            "oracle_coverage_weight": float(pretrain_oracle_coverage_weight),
        },
        "pretrain_metrics": dict(pretrain_bundle.get("metrics", {}) or {}),
        "finetune_summary": finetune_summary,
    }
    _write_json(summary, out_dir / "oracle_shared_candidate_pretrain_summary.json")
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description="Oracle pretraining pipeline for shared repair/build candidate encoder")
    p.add_argument("--spec_glob", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--seeds", type=str, default="0")
    p.add_argument("--depth_min", type=int, default=3)
    p.add_argument("--depth_max", type=int, default=8)
    p.add_argument("--topk", type=int, default=8)
    p.add_argument("--max_corrupt_paths_per_spec", type=int, default=2)
    p.add_argument("--sweep_max_paths", type=int, default=4)
    p.add_argument("--hidden_dim", type=int, default=32)
    p.add_argument("--pretrain_epochs", type=int, default=120)
    p.add_argument("--pretrain_lr", type=float, default=5.0e-3)
    p.add_argument("--pretrain_seed", type=int, default=0)
    p.add_argument("--pretrain_repair_rank_weight", type=float, default=0.15)
    p.add_argument("--pretrain_build_rank_weight", type=float, default=0.15)
    p.add_argument("--pretrain_common_rank_weight", type=float, default=0.35)
    p.add_argument("--pretrain_path_ce_weight", type=float, default=0.02)
    p.add_argument("--pretrain_common_value_weight", type=float, default=0.15)
    p.add_argument("--pretrain_preview_value_weight", type=float, default=0.02)
    p.add_argument("--pretrain_preview_regret_weight", type=float, default=0.05)
    p.add_argument("--pretrain_common_state_value_weight", type=float, default=0.02)
    p.add_argument("--pretrain_repair_state_value_weight", type=float, default=0.0)
    p.add_argument("--pretrain_build_state_value_weight", type=float, default=0.0)
    p.add_argument("--pretrain_oracle_path_weight", type=float, default=0.15)
    p.add_argument("--pretrain_oracle_relation_weight", type=float, default=0.10)
    p.add_argument("--pretrain_oracle_mode_weight", type=float, default=0.10)
    p.add_argument("--pretrain_oracle_truth_weight", type=float, default=0.10)
    p.add_argument("--pretrain_oracle_mode_best_weight", type=float, default=0.10)
    p.add_argument("--pretrain_oracle_rank_weight", type=float, default=0.15)
    p.add_argument("--pretrain_oracle_stability_weight", type=float, default=0.12)
    p.add_argument("--pretrain_oracle_coverage_weight", type=float, default=0.10)
    p.add_argument("--finetune_report_paths", nargs="*", default=[])
    p.add_argument("--finetune_epochs", type=int, default=120)
    p.add_argument("--finetune_lr", type=float, default=5.0e-3)
    p.add_argument("--finetune_seed", type=int, default=0)
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    seeds = [int(tok.strip()) for tok in str(args.seeds).split(",") if tok.strip()]
    summary = run_shared_candidate_oracle_pretrain_pipeline(
        spec_glob=str(args.spec_glob),
        output_dir=str(args.output_dir),
        seeds=seeds,
        depth_min=int(args.depth_min),
        depth_max=int(args.depth_max),
        topk=int(args.topk),
        max_corrupt_paths_per_spec=int(args.max_corrupt_paths_per_spec),
        sweep_max_paths=int(args.sweep_max_paths),
        hidden_dim=int(args.hidden_dim),
        pretrain_epochs=int(args.pretrain_epochs),
        pretrain_lr=float(args.pretrain_lr),
        pretrain_seed=int(args.pretrain_seed),
        pretrain_repair_rank_weight=float(args.pretrain_repair_rank_weight),
        pretrain_build_rank_weight=float(args.pretrain_build_rank_weight),
        pretrain_common_rank_weight=float(args.pretrain_common_rank_weight),
        pretrain_path_ce_weight=float(args.pretrain_path_ce_weight),
        pretrain_common_value_weight=float(args.pretrain_common_value_weight),
        pretrain_preview_value_weight=float(args.pretrain_preview_value_weight),
        pretrain_preview_regret_weight=float(args.pretrain_preview_regret_weight),
        pretrain_common_state_value_weight=float(args.pretrain_common_state_value_weight),
        pretrain_repair_state_value_weight=float(args.pretrain_repair_state_value_weight),
        pretrain_build_state_value_weight=float(args.pretrain_build_state_value_weight),
        pretrain_oracle_path_weight=float(args.pretrain_oracle_path_weight),
        pretrain_oracle_relation_weight=float(args.pretrain_oracle_relation_weight),
        pretrain_oracle_mode_weight=float(args.pretrain_oracle_mode_weight),
        pretrain_oracle_truth_weight=float(args.pretrain_oracle_truth_weight),
        pretrain_oracle_mode_best_weight=float(args.pretrain_oracle_mode_best_weight),
        pretrain_oracle_rank_weight=float(args.pretrain_oracle_rank_weight),
        pretrain_oracle_stability_weight=float(args.pretrain_oracle_stability_weight),
        pretrain_oracle_coverage_weight=float(args.pretrain_oracle_coverage_weight),
        finetune_report_paths=list(args.finetune_report_paths),
        finetune_epochs=int(args.finetune_epochs),
        finetune_lr=float(args.finetune_lr),
        finetune_seed=int(args.finetune_seed),
        quiet=bool(args.quiet),
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
