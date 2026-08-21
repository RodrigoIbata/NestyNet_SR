# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

from __future__ import annotations

import argparse
import json
import pathlib
import tempfile
from typing import Any, Mapping, Sequence

import torch

from nestynet_sr.sr_search.factorized_search.controller_harness import (
    DEFAULT_COMPARISON_MODE,
    DEFAULT_SEEDS as CONTROLLER_DEFAULT_SEEDS,
    DEFAULT_TARGETS as CONTROLLER_DEFAULT_TARGETS,
    run_controller_benchmark,
)
from nestynet_sr.sr_search.factorized_search.stage1_benchmark_harness import (
    DEFAULT_SEEDS as STAGE1_DEFAULT_SEEDS,
    DEFAULT_TARGETS as STAGE1_DEFAULT_TARGETS,
    run_stage1_benchmark,
)
from nestynet_sr.sr_search.factorized_search.scheduler_critic import load_scheduler_bundle
from nestynet_sr.sr_search.factorized_search.scheduler_train import run_scheduler_training


DEFAULT_ABLATIONS = (
    "default",
    "no_uncertainty_bonus",
    "no_cost_term",
    "no_fragility_stability",
    "no_hole_route",
)


def _dtype_from_name(name: str) -> torch.dtype:
    token = str(name or "float64").strip().lower()
    if token in {"float32", "f32", "fp32"}:
        return torch.float32
    if token in {"float64", "f64", "fp64", "double"}:
        return torch.float64
    raise ValueError(f"unknown dtype: {name!r}")


def _ablation_spec(name: str) -> dict[str, Any]:
    token = str(name or "default").strip().lower()
    specs: dict[str, dict[str, Any]] = {
        "default": {
            "description": "Baseline scheduler settings with advisory/control arms enabled.",
            "scheduler_arm_overrides": {},
        },
        "budget_aware": {
            "description": "Keep the full scheduler budget ladder active.",
            "scheduler_arm_overrides": {},
        },
        "one_step_budget_only": {
            "description": "Restrict the live scheduler to a single exact-budget step.",
            "scheduler_budget_ladder": [1],
            "training_variant": {
                "budget_ladder": [1],
                "optional_dataset_paths": True,
                "init_from_base": True,
            },
        },
        "preview_only_build": {
            "description": "Keep the fair preview-only build slate in scheduler comparisons.",
            "scheduler_arm_overrides": {
                "scheduler_build_preview_only": True,
            },
        },
        "exact_scored_build": {
            "description": "Allow build candidates to exact-score before scheduler choice.",
            "scheduler_arm_overrides": {
                "scheduler_build_preview_only": False,
            },
        },
        "no_uncertainty_bonus": {
            "description": "Disable the ensemble uncertainty bonus in the live chooser.",
            "scheduler_arm_overrides": {
                "scheduler_uncertainty_bonus": 0.0,
            },
        },
        "no_cost_term": {
            "description": "Zero out scheduler cost penalties while keeping the rest of the acquisition fixed.",
            "scheduler_arm_overrides": {
                "scheduler_acquisition_weights": {
                    "cost_exact": 0.0,
                    "cost_wall": 0.0,
                },
            },
        },
        "no_fragility_stability": {
            "description": "Remove fragility and stability terms from the live scheduler acquisition.",
            "scheduler_arm_overrides": {
                "scheduler_acquisition_weights": {
                    "fragile": 0.0,
                    "stable": 0.0,
                },
            },
        },
        "no_hole_route": {
            "description": "Disable hole-search opportunities so the scheduler only arbitrates build/repair.",
            "scheduler_arm_overrides": {
                "hole_search_enable": False,
                "hole_search_first_class_scheduler_enable": False,
                "hole_search_route_scheduler_enable": False,
            },
        },
        "separate_route_families": {
            "description": "Keep repair/build/hole as separate route families.",
            "scheduler_arm_overrides": {},
            "training_variant": {
                "route_aliases": {},
                "optional_dataset_paths": True,
                "init_from_base": True,
            },
        },
        "merged_repair_hole_route_families": {
            "description": "Merge hole rows into the repair route family for scheduler training and prediction.",
            "scheduler_arm_overrides": {},
            "training_variant": {
                "route_aliases": {
                    "hole": "repair",
                },
                "requires_dataset_paths": True,
                "init_from_base": True,
            },
        },
        "oracle_pretrained_only": {
            "description": "Use the provided oracle-pretrained scheduler bundle without live advisory finetuning.",
            "scheduler_arm_overrides": {},
        },
        "oracle_refine_live_advisory_finetune": {
            "description": "Finetune the provided scheduler bundle on live advisory rows before benchmarking.",
            "scheduler_arm_overrides": {},
            "training_variant": {
                "route_aliases": {},
                "requires_dataset_paths": True,
                "init_from_base": True,
            },
        },
    }
    if token not in specs:
        raise ValueError(f"unknown scheduler ablation: {name!r}")
    return {
        "name": str(token),
        **specs[token],
    }


def _resolved_budget_ladder(
    scheduler_budget_ladder: Sequence[int] | None,
    *,
    spec: Mapping[str, Any],
    base_bundle: Mapping[str, Any] | None,
) -> list[int] | None:
    explicit = spec.get("scheduler_budget_ladder", None)
    if explicit is not None:
        return [int(v) for v in list(explicit)]
    if scheduler_budget_ladder is not None:
        return [int(v) for v in list(scheduler_budget_ladder)]
    if isinstance(base_bundle, Mapping):
        bundle_ladder = base_bundle.get("budget_ladder", None)
        if bundle_ladder is not None:
            return [int(v) for v in list(bundle_ladder)]
    return None


def _prepare_ablation_bundle(
    *,
    spec: Mapping[str, Any],
    scheduler_bundle_path: str,
    scheduler_dataset_paths: Sequence[str],
    scheduler_budget_ladder: Sequence[int] | None,
    finetune_epochs: int,
    finetune_lr: float,
    finetune_weight_decay: float,
    finetune_val_fraction: float,
    finetune_seed: int,
) -> tuple[str, dict[str, Any]]:
    bundle_path = str(scheduler_bundle_path or "").strip()
    meta = {
        "bundle_source": "provided_bundle",
        "derived_bundle_path": bundle_path,
        "training_summary": None,
    }
    training_variant = dict(spec.get("training_variant", {}) or {})
    if not training_variant:
        return bundle_path, meta
    dataset_list = [str(path) for path in list(scheduler_dataset_paths or []) if str(path).strip()]
    if not dataset_list:
        if bool(training_variant.get("requires_dataset_paths", False)):
            raise ValueError(
                f"ablation {spec.get('name')!r} requires scheduler_dataset_paths for derived-bundle training"
            )
        return bundle_path, meta
    base_bundle = load_scheduler_bundle(bundle_path)
    resolved_budget_ladder = training_variant.get(
        "budget_ladder",
        _resolved_budget_ladder(scheduler_budget_ladder, spec=spec, base_bundle=base_bundle),
    )
    resolved_threshold_ladder = training_variant.get(
        "threshold_ladder",
        base_bundle.get("threshold_ladder", None),
    )
    route_aliases = dict(training_variant.get("route_aliases", {}) or {})
    tmp_dir = pathlib.Path(tempfile.mkdtemp(prefix=f"scheduler_ablation_{spec.get('name', 'variant')}_"))
    output_path = tmp_dir / "scheduler_bundle.pt"
    summary = run_scheduler_training(
        dataset_paths=list(dataset_list),
        output_path=str(output_path),
        init_bundle_path=bundle_path if bool(training_variant.get("init_from_base", True)) else "",
        budget_ladder=None if resolved_budget_ladder is None else [int(v) for v in list(resolved_budget_ladder)],
        threshold_ladder=None if resolved_threshold_ladder is None else [float(v) for v in list(resolved_threshold_ladder)],
        route_aliases={str(k): str(v) for k, v in route_aliases.items()},
        hidden_dim=int(base_bundle.get("hidden_dim", 64)),
        epochs=int(finetune_epochs),
        lr=float(finetune_lr),
        weight_decay=float(finetune_weight_decay),
        val_fraction=float(finetune_val_fraction),
        seed=int(finetune_seed),
        ensemble_size=int(base_bundle.get("ensemble_size", 1) or 1),
    )
    return str(output_path), {
        "bundle_source": "derived_bundle",
        "derived_bundle_path": str(output_path),
        "training_summary": dict(summary or {}),
    }


def _primary_comparison_key(report: Mapping[str, Any]) -> str | None:
    comparisons = dict(report.get("comparisons", {}) or {})
    for needle in ("scheduler_control", "scheduler_advisory", "macro"):
        for key in comparisons:
            if needle in str(key):
                return str(key)
    if comparisons:
        return str(next(iter(comparisons)))
    return None


def run_scheduler_ablations(
    *,
    benchmark_kind: str = "controller",
    ablations: Sequence[str] = DEFAULT_ABLATIONS,
    targets: Sequence[str] | None = None,
    seeds: Sequence[int] | None = None,
    scheduler_bundle_path: str,
    critic_path: str = "",
    scheduler_dataset_paths: Sequence[str] = (),
    scheduler_budget_ladder: Sequence[int] | None = None,
    comparison_mode: str = DEFAULT_COMPARISON_MODE,
    macro_profile: str = "default",
    n_iter: int = 120,
    max_depth: int = 5,
    brute_depth: int = 0,
    n_fit: int = 128,
    n_probe: int = 512,
    refine_enable: bool = True,
    solve_mse: float = 1.0e-10,
    finetune_epochs: int = 40,
    finetune_lr: float = 5.0e-3,
    finetune_weight_decay: float = 1.0e-4,
    finetune_val_fraction: float = 0.2,
    finetune_seed: int = 0,
    dtype: torch.dtype = torch.float64,
    threads: int | None = 1,
) -> dict[str, Any]:
    kind = str(benchmark_kind or "controller").strip().lower()
    if kind not in {"controller", "stage1"}:
        raise ValueError(f"unknown benchmark_kind: {benchmark_kind!r}")
    target_list = [str(v) for v in (targets or (CONTROLLER_DEFAULT_TARGETS if kind == "controller" else STAGE1_DEFAULT_TARGETS))]
    seed_list = [int(v) for v in (seeds or (CONTROLLER_DEFAULT_SEEDS if kind == "controller" else STAGE1_DEFAULT_SEEDS))]
    if not str(scheduler_bundle_path or "").strip():
        raise ValueError("scheduler_bundle_path is required for scheduler ablations")
    base_bundle = load_scheduler_bundle(str(scheduler_bundle_path))

    results: dict[str, Any] = {}
    summaries: dict[str, Any] = {}
    order: list[str] = []
    for raw_name in ablations:
        spec = _ablation_spec(raw_name)
        name = str(spec["name"])
        order.append(name)
        scheduler_arm_overrides = dict(spec.get("scheduler_arm_overrides", {}) or {})
        ablation_bundle_path, bundle_meta = _prepare_ablation_bundle(
            spec=spec,
            scheduler_bundle_path=str(scheduler_bundle_path),
            scheduler_dataset_paths=scheduler_dataset_paths,
            scheduler_budget_ladder=scheduler_budget_ladder,
            finetune_epochs=int(finetune_epochs),
            finetune_lr=float(finetune_lr),
            finetune_weight_decay=float(finetune_weight_decay),
            finetune_val_fraction=float(finetune_val_fraction),
            finetune_seed=int(finetune_seed),
        )
        ablation_budget_ladder = _resolved_budget_ladder(
            scheduler_budget_ladder,
            spec=spec,
            base_bundle=base_bundle,
        )
        if kind == "controller":
            report = run_controller_benchmark(
                targets=target_list,
                seeds=seed_list,
                profile=str(macro_profile),
                scheduler_bundle_path=str(ablation_bundle_path),
                scheduler_budget_ladder=ablation_budget_ladder,
                scheduler_arm_overrides=scheduler_arm_overrides,
                comparison_mode=str(comparison_mode),
                n_iter=int(n_iter),
                max_depth=int(max_depth),
                brute_depth=int(brute_depth),
                solve_mse=float(solve_mse),
                dtype=dtype,
                capture_search_output=True,
                threads=threads,
            )
        else:
            report = run_stage1_benchmark(
                targets=target_list,
                seeds=seed_list,
                critic_path=str(critic_path or ""),
                scheduler_bundle_path=str(ablation_bundle_path),
                scheduler_dataset_paths=[str(path) for path in scheduler_dataset_paths],
                scheduler_budget_ladder=ablation_budget_ladder,
                scheduler_arm_overrides=scheduler_arm_overrides,
                comparison_mode=str(comparison_mode),
                blends=(),
                arm_modes=("scheduler_advisory", "scheduler_control"),
                macro_profile=str(macro_profile),
                n_iter=int(n_iter),
                max_depth=int(max_depth),
                brute_depth=int(brute_depth),
                n_fit=int(n_fit),
                n_probe=int(n_probe),
                refine_enable=bool(refine_enable),
                solve_mse=float(solve_mse),
                dtype=dtype,
                capture_search_output=True,
                threads=threads,
            )
        primary_key = _primary_comparison_key(report)
        summaries[name] = {
            "description": str(spec.get("description", "")),
            "scheduler_arm_overrides": scheduler_arm_overrides,
            "scheduler_budget_ladder": None
            if ablation_budget_ladder is None
            else [int(v) for v in list(ablation_budget_ladder)],
            "bundle_source": str(bundle_meta.get("bundle_source", "provided_bundle")),
            "bundle_path": str(bundle_meta.get("derived_bundle_path", ablation_bundle_path)),
            "training_summary": bundle_meta.get("training_summary", None),
            "primary_comparison_key": primary_key,
            "primary_comparison": None
            if primary_key is None
            else dict((report.get("comparisons", {}) or {}).get(primary_key, {}) or {}),
        }
        results[name] = report
    return {
        "config": {
            "benchmark_kind": str(kind),
            "targets": target_list,
            "seeds": seed_list,
            "scheduler_bundle_path": str(scheduler_bundle_path),
            "critic_path": str(critic_path or ""),
            "scheduler_dataset_paths": [str(path) for path in scheduler_dataset_paths],
            "scheduler_budget_ladder": None
            if scheduler_budget_ladder is None
            else [int(v) for v in list(scheduler_budget_ladder)],
            "comparison_mode": str(comparison_mode),
            "macro_profile": str(macro_profile),
            "n_iter": int(n_iter),
            "max_depth": int(max_depth),
            "brute_depth": int(brute_depth),
            "n_fit": int(n_fit),
            "n_probe": int(n_probe),
            "refine_enable": bool(refine_enable),
            "solve_mse": float(solve_mse),
            "finetune_epochs": int(finetune_epochs),
            "finetune_lr": float(finetune_lr),
            "finetune_weight_decay": float(finetune_weight_decay),
            "finetune_val_fraction": float(finetune_val_fraction),
            "finetune_seed": int(finetune_seed),
            "dtype": str(dtype),
            "threads": None if threads is None else int(threads),
        },
        "ablation_order": order,
        "summaries": summaries,
        "results": results,
    }


def _print_human_report(report: Mapping[str, Any]) -> None:
    cfg = dict(report.get("config", {}) or {})
    print("\n=== Scheduler ablations ===")
    print(
        f"benchmark={cfg.get('benchmark_kind')} targets={cfg.get('targets')} "
        f"seeds={cfg.get('seeds')} comparison_mode={cfg.get('comparison_mode')}"
    )
    for name in list(report.get("ablation_order", []) or []):
        summary = dict((report.get("summaries", {}) or {}).get(name, {}) or {})
        primary = dict(summary.get("primary_comparison", {}) or {})
        print(
            f"  {name}: key={summary.get('primary_comparison_key')} "
            f"matched_win_rate={primary.get('matched_win_rate')} "
            f"mean_matched_delta_log_eff={primary.get('mean_matched_delta_log_eff')} "
            f"mean_cost_ratio={primary.get('mean_candidate_cost_ratio')}"
        )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--benchmark_kind", type=str, default="controller", choices=["controller", "stage1"])
    p.add_argument("--ablations", nargs="*", default=list(DEFAULT_ABLATIONS))
    p.add_argument("--targets", nargs="*", default=None)
    p.add_argument("--seeds", nargs="*", type=int, default=None)
    p.add_argument("--scheduler_bundle_path", type=str, required=True)
    p.add_argument("--critic_path", type=str, default="")
    p.add_argument("--scheduler_dataset_paths", nargs="*", default=[])
    p.add_argument("--scheduler_budget_ladder", type=str, default="1,2,4,8")
    p.add_argument("--comparison_mode", type=str, default=DEFAULT_COMPARISON_MODE, choices=["raw", "matched_exact", "matched_wall"])
    p.add_argument("--macro_profile", type=str, default="default", choices=["default", "repair_probe"])
    p.add_argument("--n_iter", type=int, default=120)
    p.add_argument("--max_depth", type=int, default=5)
    p.add_argument("--brute_depth", type=int, default=0)
    p.add_argument("--n_fit", type=int, default=128)
    p.add_argument("--n_probe", type=int, default=512)
    p.add_argument("--solve_mse", type=float, default=1.0e-10)
    p.add_argument("--finetune_epochs", type=int, default=40)
    p.add_argument("--finetune_lr", type=float, default=5.0e-3)
    p.add_argument("--finetune_weight_decay", type=float, default=1.0e-4)
    p.add_argument("--finetune_val_fraction", type=float, default=0.2)
    p.add_argument("--finetune_seed", type=int, default=0)
    p.add_argument("--dtype", type=str, default="float64", choices=["float32", "float64"])
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--no_plus", action="store_true")
    p.add_argument("--json", action="store_true")
    p.add_argument("--output", type=str, default="")
    args = p.parse_args()

    report = run_scheduler_ablations(
        benchmark_kind=str(args.benchmark_kind),
        ablations=[str(name) for name in args.ablations],
        targets=None if args.targets is None else [str(v) for v in args.targets],
        seeds=None if args.seeds is None else [int(v) for v in args.seeds],
        scheduler_bundle_path=str(args.scheduler_bundle_path),
        critic_path=str(args.critic_path),
        scheduler_dataset_paths=[str(path) for path in args.scheduler_dataset_paths],
        scheduler_budget_ladder=[
            int(token.strip())
            for token in str(args.scheduler_budget_ladder).split(",")
            if token.strip()
        ],
        comparison_mode=str(args.comparison_mode),
        macro_profile=str(args.macro_profile),
        n_iter=int(args.n_iter),
        max_depth=int(args.max_depth),
        brute_depth=int(args.brute_depth),
        n_fit=int(args.n_fit),
        n_probe=int(args.n_probe),
        refine_enable=not bool(args.no_plus),
        solve_mse=float(args.solve_mse),
        finetune_epochs=int(args.finetune_epochs),
        finetune_lr=float(args.finetune_lr),
        finetune_weight_decay=float(args.finetune_weight_decay),
        finetune_val_fraction=float(args.finetune_val_fraction),
        finetune_seed=int(args.finetune_seed),
        dtype=_dtype_from_name(args.dtype),
        threads=args.threads,
    )

    if args.output:
        out_path = pathlib.Path(str(args.output))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    if bool(args.json):
        print(json.dumps(report, indent=2))
    else:
        _print_human_report(report)


if __name__ == "__main__":
    main()
