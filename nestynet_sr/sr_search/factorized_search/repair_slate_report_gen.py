# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

from __future__ import annotations

import argparse
import contextlib
import io
import json
import time
from pathlib import Path

import torch

from nestynet_sr.sr_search.factorized_search import explorer
from nestynet_sr.sr_search.factorized_search.controller_harness import (
    _profile_overrides,
    _resolve_target_spec,
    _set_torch_threads,
)
from nestynet_sr.sr_search.factorized_search.engine.search import run_explorer_core


def generate_repair_slate_report(
    *,
    target: str,
    seed: int,
    output_path: str,
    profile: str = "repair_probe",
    n_iter: int = 120,
    max_depth: int = 5,
    brute_depth: int = 0,
    n_fit: int = 128,
    n_probe: int = 512,
    refine_enable: bool = True,
    inverse_max_paths: int = 12,
    controller_build_slate_enable: bool = False,
    controller_build_slate_actions: tuple[str, ...] = ("replace", "wrap_un", "residual"),
    controller_build_slate_max_actions: int = 3,
    repair_controller_credible_route_enable: bool = False,
    repair_opportunity_controller_path: str = "",
    repair_controller_route_compare_enable: bool = False,
    repair_controller_route_compare_path: str = "",
    repair_controller_route_compare_repair_tuple_path: str = "",
    repair_controller_route_compare_build_tuple_path: str = "",
    repair_controller_route_compare_max_repair_prob: float = 0.35,
    repair_controller_route_compare_min_build_margin: float = 0.05,
    repair_controller_max_setup_steps: int = 0,
    repair_controller_setup_step_value_min: float = 0.10,
    repair_controller_setup_step_regret_max: float = 0.50,
    repair_controller_setup_step_max_worsen: float = 0.05,
    threads: int = 1,
) -> dict:
    _set_torch_threads(max(1, int(threads)))
    spec = _resolve_target_spec(str(target))
    kwargs = {
        "target_fn": spec["fn"],
        "nvars": int(spec["nvars"]),
        "seed": int(seed),
        "n_iter": int(n_iter),
        "max_depth": int(max_depth),
        "brute_depth": int(brute_depth),
        "n_fit": int(n_fit),
        "n_probe": int(n_probe),
        "dtype": torch.float64,
        "print_every": 0,
        "refine_enable": bool(refine_enable),
        "inverse_max_paths": int(inverse_max_paths),
        "controller_build_slate_enable": bool(controller_build_slate_enable),
        "controller_build_slate_actions": list(controller_build_slate_actions),
        "controller_build_slate_max_actions": int(controller_build_slate_max_actions),
        "repair_controller_credible_route_enable": bool(repair_controller_credible_route_enable),
        "repair_opportunity_controller_path": str(repair_opportunity_controller_path or ""),
        "repair_controller_route_compare_enable": bool(repair_controller_route_compare_enable),
        "repair_controller_route_compare_path": str(repair_controller_route_compare_path or ""),
        "repair_controller_route_compare_repair_tuple_path": str(repair_controller_route_compare_repair_tuple_path or ""),
        "repair_controller_route_compare_build_tuple_path": str(repair_controller_route_compare_build_tuple_path or ""),
        "repair_controller_route_compare_max_repair_prob": float(repair_controller_route_compare_max_repair_prob),
        "repair_controller_route_compare_min_build_margin": float(repair_controller_route_compare_min_build_margin),
        "repair_controller_max_setup_steps": int(repair_controller_max_setup_steps),
        "repair_controller_setup_step_value_min": float(repair_controller_setup_step_value_min),
        "repair_controller_setup_step_regret_max": float(repair_controller_setup_step_regret_max),
        "repair_controller_setup_step_max_worsen": float(repair_controller_setup_step_max_worsen),
        "repair_controller_critic_enable": False,
        "lo": spec.get("lo", 1.0),
        "hi": spec.get("hi", 5.0),
        "y_dims": spec.get("y_dims", None),
        "var_dims": spec.get("var_dims", None),
    }
    kwargs.update(_profile_overrides(str(profile), macro_enabled=False))
    kwargs.setdefault("_runtime_hooks", explorer.make_engine_runtime_hooks())

    buf = io.StringIO()
    start = time.time()
    with contextlib.redirect_stdout(buf):
        arch = run_explorer_core(**kwargs)
    elapsed_s = time.time() - start

    inv_log = list(getattr(arch, "inverse_experiment_log", []) or [])
    best_mse = float("inf")
    best_expr = ""
    try:
        best = arch.best(1)[0]
        best_mse = float(getattr(best, "best_mse", float("inf")))
        best_expr = str(explorer.node_str(getattr(best, "best_expr", None)))
    except Exception:
        pass

    payload = {
        "target": str(target),
        "seed": int(seed),
        "profile": str(profile),
        "kwargs": {
            "n_iter": int(n_iter),
            "max_depth": int(max_depth),
            "n_fit": int(n_fit),
            "n_probe": int(n_probe),
            "refine_enable": bool(refine_enable),
            "inverse_max_paths": int(inverse_max_paths),
            "controller_build_slate_enable": bool(controller_build_slate_enable),
            "controller_build_slate_actions": list(controller_build_slate_actions),
            "controller_build_slate_max_actions": int(controller_build_slate_max_actions),
            "repair_controller_credible_route_enable": bool(repair_controller_credible_route_enable),
            "repair_opportunity_controller_path": str(repair_opportunity_controller_path or ""),
            "repair_controller_route_compare_enable": bool(repair_controller_route_compare_enable),
            "repair_controller_route_compare_path": str(repair_controller_route_compare_path or ""),
            "repair_controller_route_compare_repair_tuple_path": str(repair_controller_route_compare_repair_tuple_path or ""),
            "repair_controller_route_compare_build_tuple_path": str(repair_controller_route_compare_build_tuple_path or ""),
            "repair_controller_route_compare_max_repair_prob": float(repair_controller_route_compare_max_repair_prob),
            "repair_controller_route_compare_min_build_margin": float(repair_controller_route_compare_min_build_margin),
            "repair_controller_max_setup_steps": int(repair_controller_max_setup_steps),
            "repair_controller_setup_step_value_min": float(repair_controller_setup_step_value_min),
            "repair_controller_setup_step_regret_max": float(repair_controller_setup_step_regret_max),
            "repair_controller_setup_step_max_worsen": float(repair_controller_setup_step_max_worsen),
        },
        "best_mse": float(best_mse),
        "best_expr": str(best_expr),
        "residual_basins": int(len(getattr(arch, "d", {}) or {})),
        "n_eval": int(getattr(arch, "n_eval", 0)),
        "elapsed_s": float(elapsed_s),
        "inverse_experiment_log": inv_log,
        "search_stdout_tail": [str(line) for line in buf.getvalue().splitlines()[-8:] if str(line).strip()],
    }
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--target", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--output_path", required=True)
    p.add_argument("--profile", default="repair_probe")
    p.add_argument("--n_iter", type=int, default=120)
    p.add_argument("--max_depth", type=int, default=5)
    p.add_argument("--brute_depth", type=int, default=0)
    p.add_argument("--n_fit", type=int, default=128)
    p.add_argument("--n_probe", type=int, default=512)
    p.add_argument("--inverse_max_paths", type=int, default=12)
    p.add_argument("--controller_build_slate_enable", action="store_true")
    p.add_argument("--controller_build_slate_actions", type=str, default="replace,wrap_un,residual")
    p.add_argument("--controller_build_slate_max_actions", type=int, default=3)
    p.add_argument("--repair_controller_credible_route_enable", action="store_true")
    p.add_argument("--repair_opportunity_controller_path", type=str, default="")
    p.add_argument("--repair_controller_route_compare_enable", action="store_true")
    p.add_argument("--repair_controller_route_compare_path", type=str, default="")
    p.add_argument("--repair_controller_route_compare_repair_tuple_path", type=str, default="")
    p.add_argument("--repair_controller_route_compare_build_tuple_path", type=str, default="")
    p.add_argument("--repair_controller_route_compare_max_repair_prob", type=float, default=0.35)
    p.add_argument("--repair_controller_route_compare_min_build_margin", type=float, default=0.05)
    p.add_argument("--repair_controller_max_setup_steps", type=int, default=0)
    p.add_argument("--repair_controller_setup_step_value_min", type=float, default=0.10)
    p.add_argument("--repair_controller_setup_step_regret_max", type=float, default=0.50)
    p.add_argument("--repair_controller_setup_step_max_worsen", type=float, default=0.05)
    p.add_argument("--threads", type=int, default=1)
    args = p.parse_args()

    payload = generate_repair_slate_report(
        target=str(args.target),
        seed=int(args.seed),
        output_path=str(args.output_path),
        profile=str(args.profile),
        n_iter=int(args.n_iter),
        max_depth=int(args.max_depth),
        brute_depth=int(args.brute_depth),
        n_fit=int(args.n_fit),
        n_probe=int(args.n_probe),
        inverse_max_paths=int(args.inverse_max_paths),
        controller_build_slate_enable=bool(args.controller_build_slate_enable),
        controller_build_slate_actions=tuple(str(args.controller_build_slate_actions).split(",")),
        controller_build_slate_max_actions=int(args.controller_build_slate_max_actions),
        repair_controller_credible_route_enable=bool(args.repair_controller_credible_route_enable),
        repair_opportunity_controller_path=str(args.repair_opportunity_controller_path),
        repair_controller_route_compare_enable=bool(args.repair_controller_route_compare_enable),
        repair_controller_route_compare_path=str(args.repair_controller_route_compare_path),
        repair_controller_route_compare_repair_tuple_path=str(args.repair_controller_route_compare_repair_tuple_path),
        repair_controller_route_compare_build_tuple_path=str(args.repair_controller_route_compare_build_tuple_path),
        repair_controller_route_compare_max_repair_prob=float(args.repair_controller_route_compare_max_repair_prob),
        repair_controller_route_compare_min_build_margin=float(args.repair_controller_route_compare_min_build_margin),
        repair_controller_max_setup_steps=int(args.repair_controller_max_setup_steps),
        repair_controller_setup_step_value_min=float(args.repair_controller_setup_step_value_min),
        repair_controller_setup_step_regret_max=float(args.repair_controller_setup_step_regret_max),
        repair_controller_setup_step_max_worsen=float(args.repair_controller_setup_step_max_worsen),
        threads=int(args.threads),
    )
    print(
        json.dumps(
            {
                "target": payload["target"],
                "seed": payload["seed"],
                "output_path": str(args.output_path),
                "n_rows": len(payload["inverse_experiment_log"]),
                "best_mse": payload["best_mse"],
                "elapsed_s": payload["elapsed_s"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
