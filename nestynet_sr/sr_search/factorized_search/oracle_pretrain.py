# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

from __future__ import annotations

import argparse
import json
import pathlib
from typing import Any, Sequence

import torch

from .config import FactorizedSearchConfig
from .oracle_lab import (
    default_oracle_hyperparams,
    generate_oracle_policy_pretrain_dataset,
)
from .oracle_suite import _resolve_spec_paths
from .repair_critic import (
    load_inverse_experiment_rows,
    pretrain_repair_controller_from_oracle_tasks,
    save_repair_critic_bundle,
    train_repair_critic,
)


def _parse_int_csv(raw: str | None, *, default: Sequence[int]) -> list[int]:
    if raw is None:
        return [int(v) for v in default]
    out: list[int] = []
    for tok in str(raw).split(","):
        t = tok.strip()
        if not t:
            continue
        out.append(int(t))
    if not out:
        return [int(v) for v in default]
    return out


def _parse_str_csv(raw: str | None, *, default: Sequence[str]) -> list[str]:
    if raw is None:
        return [str(v) for v in default]
    out = [str(tok).strip() for tok in str(raw).split(",") if str(tok).strip()]
    if not out:
        return [str(v) for v in default]
    return out


def _dtype_from_name(name: str) -> torch.dtype:
    token = str(name or "float64").strip().lower()
    if token in ("float32", "fp32", "f32"):
        return torch.float32
    if token in ("float64", "fp64", "f64", "double"):
        return torch.float64
    raise ValueError(f"unknown dtype: {name!r}")


def _write_json(payload: dict[str, Any], path: str | pathlib.Path) -> None:
    out_path = pathlib.Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _make_hp(args: argparse.Namespace) -> FactorizedSearchConfig:
    hp = default_oracle_hyperparams()
    if args.max_depth is not None:
        hp.max_depth = int(args.max_depth)
    if args.poly_degree is not None:
        hp.poly_degree = int(args.poly_degree)
    if args.n_fit is not None:
        hp.n_fit = int(args.n_fit)
    if args.n_probe is not None:
        hp.n_probe = int(args.n_probe)
    if args.inverse_max_paths is not None:
        hp.inverse_max_paths = int(args.inverse_max_paths)
    if args.inverse_topk_terms is not None:
        hp.inverse_topk_terms = int(args.inverse_topk_terms)
    if args.inverse_shortlist_mult is not None:
        hp.inverse_shortlist_mult = int(args.inverse_shortlist_mult)
    if args.inverse_min_valid_frac is not None:
        hp.inverse_min_valid_frac = float(args.inverse_min_valid_frac)
    if args.inverse_min_confidence is not None:
        hp.inverse_min_confidence = float(args.inverse_min_confidence)
    if args.inverse_confidence_mode is not None:
        hp.inverse_confidence_mode = str(args.inverse_confidence_mode)
    if args.inverse_confidence_target_gain is not None:
        hp.inverse_confidence_target_gain = float(args.inverse_confidence_target_gain)
    if args.inverse_confidence_floor is not None:
        hp.inverse_confidence_floor = float(args.inverse_confidence_floor)
    if args.inverse_branch_beam_width is not None:
        hp.inverse_branch_beam_width = int(args.inverse_branch_beam_width)
    if args.inverse_local_score_mode is not None:
        hp.inverse_local_score_mode = str(args.inverse_local_score_mode)
    if args.inverse_target_mode is not None:
        hp.inverse_target_mode = str(args.inverse_target_mode)
    if args.inverse_full_mapping_penalty is not None:
        hp.inverse_full_mapping_penalty = float(args.inverse_full_mapping_penalty)
    if args.inverse_exact_simple_target_bonus is not None:
        hp.inverse_exact_simple_target_bonus = float(args.inverse_exact_simple_target_bonus)
    if args.inverse_additive_descend_penalty is not None:
        hp.inverse_additive_descend_penalty = float(args.inverse_additive_descend_penalty)
    if args.inverse_nonadditive_leaf_penalty is not None:
        hp.inverse_nonadditive_leaf_penalty = float(args.inverse_nonadditive_leaf_penalty)
    if args.inverse_micro_search_enable is not None:
        hp.inverse_micro_search_enable = bool(args.inverse_micro_search_enable)
    if args.inverse_micro_search_max_depth is not None:
        hp.inverse_micro_search_max_depth = int(args.inverse_micro_search_max_depth)
    if args.inverse_micro_search_beam_width is not None:
        hp.inverse_micro_search_beam_width = int(args.inverse_micro_search_beam_width)
    if args.inverse_micro_search_topk is not None:
        hp.inverse_micro_search_topk = int(args.inverse_micro_search_topk)
    if args.inverse_micro_search_seed_terms is not None:
        hp.inverse_micro_search_seed_terms = int(args.inverse_micro_search_seed_terms)
    return hp


def run_oracle_pretrain_pipeline(
    spec_paths: Sequence[str | pathlib.Path],
    *,
    output_dir: str | pathlib.Path,
    factorized_search_hp: FactorizedSearchConfig | None = None,
    seeds: Sequence[int] = (0,),
    dtype: torch.dtype = torch.float64,
    enforce_dims: bool = True,
    depth_min: int = 3,
    depth_max: int = 8,
    compare_modes: Sequence[str] = ("identity", "full", "affine"),
    topk: int = 8,
    max_corrupt_paths_per_spec: int | None = None,
    sweep_all_paths: bool = False,
    sweep_max_paths: int | None = None,
    hidden_dim: int = 32,
    pretrain_epochs: int = 200,
    pretrain_lr: float = 1.0e-2,
    pretrain_weight_decay: float = 1.0e-4,
    pretrain_val_fraction: float = 0.2,
    pretrain_seed: int = 0,
    aux_report_paths: Sequence[str | pathlib.Path] = (),
    aux_hidden_dim: int | None = None,
    aux_epochs: int = 250,
    aux_lr: float = 1.0e-2,
    aux_weight_decay: float = 1.0e-4,
    aux_val_fraction: float = 0.2,
    aux_seed: int = 0,
    continue_on_error: bool = True,
    verbose: bool = True,
) -> dict[str, Any]:
    hp = factorized_search_hp if factorized_search_hp is not None else default_oracle_hyperparams()
    out_dir = pathlib.Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    skipped_specs: list[dict[str, str]] = []
    spec_count = 0
    base_config: dict[str, Any] | None = None
    for spec_path in spec_paths:
        spec_count += 1
        try:
            payload = generate_oracle_policy_pretrain_dataset(
                [spec_path],
                factorized_search_hp=hp,
                seeds=seeds,
                dtype=dtype,
                enforce_dims=enforce_dims,
                depth_min=depth_min,
                depth_max=depth_max,
                compare_modes=compare_modes,
                topk=topk,
                max_corrupt_paths_per_spec=max_corrupt_paths_per_spec,
                sweep_all_paths=sweep_all_paths,
                sweep_max_paths=sweep_max_paths,
                verbose=verbose,
            )
        except Exception as exc:
            if not bool(continue_on_error):
                raise
            skipped_specs.append({
                "spec_path": str(spec_path),
                "error": str(exc),
            })
            continue
        if base_config is None:
            base_config = dict(payload.get("config", {}) or {})
        rows.extend(list(payload.get("rows", []) or []))
    dataset = {
        "mode": "oracle_policy_pretrain_dataset",
        "n_rows": int(len(rows)),
        "config": dict(base_config or {
            "seeds": [int(s) for s in seeds],
            "depth_min": int(depth_min),
            "depth_max": int(depth_max),
            "topk": int(topk),
            "compare_modes": [str(m) for m in compare_modes],
            "max_corrupt_paths_per_spec": None if max_corrupt_paths_per_spec is None else int(max_corrupt_paths_per_spec),
            "sweep_all_paths": bool(sweep_all_paths),
            "sweep_max_paths": None if sweep_max_paths is None else int(sweep_max_paths),
        }),
        "rows": rows,
        "skipped_specs": skipped_specs,
    }
    dataset_path = out_dir / "oracle_policy_pretrain_dataset.json"
    _write_json(dataset, dataset_path)

    pretrain_bundle = pretrain_repair_controller_from_oracle_tasks(
        dataset.get("rows", []),
        hidden_dim=int(hidden_dim),
        epochs=int(pretrain_epochs),
        lr=float(pretrain_lr),
        weight_decay=float(pretrain_weight_decay),
        val_fraction=float(pretrain_val_fraction),
        seed=int(pretrain_seed),
    )
    pretrain_bundle_path = out_dir / "oracle_policy_pretrain_bundle.pt"
    save_repair_critic_bundle(pretrain_bundle, pretrain_bundle_path)

    final_bundle = pretrain_bundle
    final_bundle_path = pretrain_bundle_path
    aux_rows = []
    if aux_report_paths:
        aux_rows = load_inverse_experiment_rows(aux_report_paths)
        if aux_rows:
            final_bundle = train_repair_critic(
                aux_rows,
                hidden_dim=int(aux_hidden_dim if aux_hidden_dim is not None else hidden_dim),
                epochs=int(aux_epochs),
                lr=float(aux_lr),
                weight_decay=float(aux_weight_decay),
                val_fraction=float(aux_val_fraction),
                seed=int(aux_seed),
                init_bundle=pretrain_bundle,
            )
            final_bundle_path = out_dir / "oracle_policy_pretrain_finetuned_bundle.pt"
            save_repair_critic_bundle(final_bundle, final_bundle_path)

    payload = {
        "output_dir": str(out_dir),
        "dataset_path": str(dataset_path),
        "pretrain_bundle_path": str(pretrain_bundle_path),
        "final_bundle_path": str(final_bundle_path),
        "n_specs": int(spec_count),
        "n_curriculum_rows": int(dataset.get("n_rows", 0)),
        "n_aux_rows": int(len(aux_rows)),
        "n_skipped_specs": int(len(skipped_specs)),
        "skipped_specs": skipped_specs,
        "pretrain_metrics": dict(pretrain_bundle.get("metrics", {}) or {}),
        "final_metrics": dict(final_bundle.get("metrics", {}) or {}),
    }
    _write_json(payload, out_dir / "oracle_pretrain_summary.json")
    return payload


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Offline oracle curriculum pretrainer for factorized symbolic search controller bundle")
    p.add_argument("--specs", nargs="*", default=None, help="Explicit equation spec files")
    p.add_argument("--spec_glob", type=str, default=None, help="Glob for equation specs")
    p.add_argument("--output_dir", type=str, default="results/oracle_pretrain")
    p.add_argument("--dtype", type=str, choices=["float32", "float64"], default="float64")
    p.add_argument("--ignore_dims", action="store_true", help="Disable dimensional filtering")
    p.add_argument("--seeds", type=str, default="0", help="Comma-separated oracle dataset seeds")
    p.add_argument("--depth_min", type=int, default=3)
    p.add_argument("--depth_max", type=int, default=8)
    p.add_argument("--compare_modes", type=str, default="identity,full,affine")
    p.add_argument("--topk", type=int, default=8)
    p.add_argument("--max_corrupt_paths_per_spec", type=int, default=None)
    p.add_argument("--sweep_all_paths", action="store_true")
    p.add_argument("--sweep_max_paths", type=int, default=None)
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--strict", action="store_true", help="Fail on the first unsupported/bad spec instead of skipping it")

    p.add_argument("--hidden_dim", type=int, default=32)
    p.add_argument("--pretrain_epochs", type=int, default=200)
    p.add_argument("--pretrain_lr", type=float, default=1.0e-2)
    p.add_argument("--pretrain_weight_decay", type=float, default=1.0e-4)
    p.add_argument("--pretrain_val_fraction", type=float, default=0.2)
    p.add_argument("--pretrain_seed", type=int, default=0)

    p.add_argument("--aux_reports", nargs="*", default=None, help="JSON reports with inverse_experiment_log rows")
    p.add_argument("--aux_hidden_dim", type=int, default=None)
    p.add_argument("--aux_epochs", type=int, default=250)
    p.add_argument("--aux_lr", type=float, default=1.0e-2)
    p.add_argument("--aux_weight_decay", type=float, default=1.0e-4)
    p.add_argument("--aux_val_fraction", type=float, default=0.2)
    p.add_argument("--aux_seed", type=int, default=0)

    p.add_argument("--max_depth", type=int, default=None)
    p.add_argument("--poly_degree", type=int, default=None)
    p.add_argument("--n_fit", type=int, default=None)
    p.add_argument("--n_probe", type=int, default=None)
    p.add_argument("--inverse_max_paths", type=int, default=None)
    p.add_argument("--inverse_topk_terms", type=int, default=None)
    p.add_argument("--inverse_shortlist_mult", type=int, default=None)
    p.add_argument("--inverse_min_valid_frac", type=float, default=None)
    p.add_argument("--inverse_min_confidence", type=float, default=None)
    p.add_argument("--inverse_confidence_mode", type=str, choices=["conditioning", "heuristic"], default=None)
    p.add_argument("--inverse_confidence_target_gain", type=float, default=None)
    p.add_argument("--inverse_confidence_floor", type=float, default=None)
    p.add_argument("--inverse_branch_beam_width", type=int, default=None)
    p.add_argument("--inverse_local_score_mode", type=str, choices=["strict", "affine", "fitbest"], default=None)
    p.add_argument("--inverse_target_mode", type=str, choices=["robust", "full", "identity", "affine", "simple"], default=None)
    p.add_argument("--inverse_full_mapping_penalty", type=float, default=None)
    p.add_argument("--inverse_exact_simple_target_bonus", type=float, default=None)
    p.add_argument("--inverse_additive_descend_penalty", type=float, default=None)
    p.add_argument("--inverse_nonadditive_leaf_penalty", type=float, default=None)
    micro_g = p.add_mutually_exclusive_group()
    micro_g.add_argument("--inverse_micro_search", dest="inverse_micro_search_enable", action="store_true")
    micro_g.add_argument("--no_inverse_micro_search", dest="inverse_micro_search_enable", action="store_false")
    p.set_defaults(inverse_micro_search_enable=None)
    p.add_argument("--inverse_micro_search_max_depth", type=int, default=None)
    p.add_argument("--inverse_micro_search_beam_width", type=int, default=None)
    p.add_argument("--inverse_micro_search_topk", type=int, default=None)
    p.add_argument("--inverse_micro_search_seed_terms", type=int, default=None)
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    spec_paths = _resolve_spec_paths(args.specs, args.spec_glob)
    hp = _make_hp(args)
    payload = run_oracle_pretrain_pipeline(
        spec_paths,
        output_dir=args.output_dir,
        factorized_search_hp=hp,
        seeds=_parse_int_csv(args.seeds, default=(0,)),
        dtype=_dtype_from_name(args.dtype),
        enforce_dims=not bool(args.ignore_dims),
        depth_min=int(args.depth_min),
        depth_max=int(args.depth_max),
        compare_modes=_parse_str_csv(args.compare_modes, default=("identity", "full", "affine")),
        topk=int(args.topk),
        max_corrupt_paths_per_spec=None if args.max_corrupt_paths_per_spec is None else int(args.max_corrupt_paths_per_spec),
        sweep_all_paths=bool(args.sweep_all_paths),
        sweep_max_paths=None if args.sweep_max_paths is None else int(args.sweep_max_paths),
        hidden_dim=int(args.hidden_dim),
        pretrain_epochs=int(args.pretrain_epochs),
        pretrain_lr=float(args.pretrain_lr),
        pretrain_weight_decay=float(args.pretrain_weight_decay),
        pretrain_val_fraction=float(args.pretrain_val_fraction),
        pretrain_seed=int(args.pretrain_seed),
        aux_report_paths=list(args.aux_reports or []),
        aux_hidden_dim=None if args.aux_hidden_dim is None else int(args.aux_hidden_dim),
        aux_epochs=int(args.aux_epochs),
        aux_lr=float(args.aux_lr),
        aux_weight_decay=float(args.aux_weight_decay),
        aux_val_fraction=float(args.aux_val_fraction),
        aux_seed=int(args.aux_seed),
        continue_on_error=not bool(args.strict),
        verbose=not bool(args.quiet),
    )
    print(
        f"[oracle-pretrain] rows={int(payload['n_curriculum_rows'])} "
        f"bundle={payload['final_bundle_path']}"
    )


if __name__ == "__main__":  # pragma: no cover
    main()
