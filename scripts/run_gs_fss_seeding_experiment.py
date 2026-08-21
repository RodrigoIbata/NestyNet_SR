#!/usr/bin/env python3
# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata

"""Engine-level FSS seeding from symmetry-reduction whole-law proposals.

Seeds discovered by ``sr_gs.de_reduction`` (pulled-back rows + whole laws,
derived from the trajectory ensemble alone) are injected into the factorized
symbolic search additive-combo pool via ``DELabSpec.extra`` and compared
against the identical unseeded search at the same (deliberately constrained)
mutation budget.  The composer's joint linear solve assembles and
coefficient-fits laws from the pinned seed atoms, so a correct seed can
short-circuit the search in the pre-mutation phase.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from nestynet_sr.sr_search.factorized_search.oracle_lab_de import (  # noqa: E402
    DEFeatureTensors,
    DELabSpec,
    default_oracle_de_hyperparams,
    run_oracle_de_from_features,
)
from nestynet_sr.sr_gs.de_reduction import symmetry_reduction_proposals  # noqa: E402


def _case_900(n: int = 300):
    x = np.linspace(0.0, 3.0, n)
    trajs, u1s = [], []
    for c in (0.7, 1.3, 2.1, 0.9, 1.6, 1.1):
        u = c / (1.0 + x)
        trajs.append((x, u))
        u1s.append(-c / (1.0 + x) ** 2)
    return "de900 u' = -u/(1+x)", trajs, u1s


def _case_903(n: int = 300):
    x = np.linspace(0.0, 3.0, n)
    trajs, u1s = [], []
    for u0 in (-0.2, 0.4, 0.9, 0.1, 0.6, -0.4):
        u = -np.log(np.exp(-u0) + x)
        trajs.append((x, u))
        u1s.append(-np.exp(u))
    return "de903 u' = -exp(u)", trajs, u1s


def _features(trajs, u1s, n_fit: int) -> DEFeatureTensors:
    def _pool(items):
        return torch.as_tensor(np.concatenate(items), dtype=torch.float64).reshape(-1, 1)

    x_fit = _pool([t[0] for t in trajs[:n_fit]])
    u_fit = _pool([t[1] for t in trajs[:n_fit]])
    du_fit = _pool(list(u1s[:n_fit]))
    x_probe = _pool([t[0] for t in trajs[n_fit:]])
    u_probe = _pool([t[1] for t in trajs[n_fit:]])
    du_probe = _pool(list(u1s[n_fit:]))
    zeros_f = torch.zeros_like(du_fit)
    zeros_p = torch.zeros_like(du_probe)
    return DEFeatureTensors(
        x_fit=x_fit, u_fit=u_fit, du_fit=du_fit, d2u_fit=zeros_f,
        x_probe=x_probe, u_probe=u_probe, du_probe=du_probe, d2u_probe=zeros_p,
    )


def _seed_payload(trajs, u1s, n_fit: int) -> list[dict[str, Any]]:
    result = symmetry_reduction_proposals(trajs[:n_fit], u1_list=u1s[:n_fit], order=1)
    payload: list[dict[str, Any]] = []
    for term, _source, family in result.get("library_rows", []):
        payload.append({"node": term, "label": f"gs_row:{family}"})
    for prop in result.get("proposals", []):
        rhs = prop.get("rhs_ast")
        if rhs is not None:
            payload.append({"node": rhs, "label": f"gs_law:{prop.get('fit_family', '')}"})
    return payload


def _fresh_hp(n_iter: int):
    hp = default_oracle_de_hyperparams()
    hp.n_iter = int(n_iter)
    hp.max_depth = 5
    hp.return_topk = 6
    hp.de_sparse_combo_enable = True
    hp.de_sparse_combo_max_terms = 3
    hp.de_sparse_combo_pool_topk = 8
    # trim expensive refinement so the seeded-vs-unseeded structural contrast is
    # what varies (not LBFGS polish); keeps the whole experiment to a few minutes
    hp.refine_num_restarts = 1
    hp.refine_final_polish = False
    hp.refine_max_trials = 4
    return hp


def _summarize(result: dict[str, Any]) -> dict[str, Any]:
    per_order = list(result.get("per_order") or [])
    entry = next(
        (po for po in per_order if int(po.get("order", -1)) == 1),
        per_order[0] if per_order else {},
    )
    best = entry.get("best") or result.get("best") or {}
    results = list(entry.get("results") or [])
    additive = entry.get("additive_fss")
    return {
        "best_expr": best.get("expr"),
        "best_score": best.get("score"),
        "best_mse": best.get("mse", best.get("mse_pooled")),
        "best_final_validated_mse": best.get("final_validated_mse"),
        "best_size": best.get("size"),
        "n_rows": len(results),
        "additive_fss": additive,
    }


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.3e}"
    except (TypeError, ValueError):
        return str(value)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Seeded vs unseeded factorized-search DE runs (engine-level symmetry seeding)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--n_iter", type=int, default=4000)
    parser.add_argument("--n_fit_traj", type=int, default=4)
    parser.add_argument("--results_dir", default=str(REPO_ROOT / "results" / "gs_fss_seeding_experiment"))
    args = parser.parse_args()

    report = []
    for case_fn in (_case_900, _case_903):
        desc, trajs, u1s = case_fn()
        feats = _features(trajs, u1s, int(args.n_fit_traj))
        payload = _seed_payload(trajs, u1s, int(args.n_fit_traj))
        entry: dict[str, Any] = {"case": desc, "n_seed_asts": len(payload),
                                 "seed_labels": [p["label"] for p in payload]}
        print(f"\n=== {desc} ===")
        print(f"  seeds from reduction: {[p['label'] for p in payload]}")
        for variant, extra in (("unseeded", None), ("seeded", {"gs_symmetry_seed_asts": tuple(payload)})):
            spec = DELabSpec(
                id=f"seeding_{variant}", csv_paths=(), order_candidates=(1,),
                include_du=False, extra=extra,
            )
            t0 = time.perf_counter()
            result = run_oracle_de_from_features(
                spec, feats, factorized_search_hp=_fresh_hp(int(args.n_iter)),
                seed=0, enforce_dims=False, verbose=False, parallel_orders=False,
            )
            wall = time.perf_counter() - t0
            summary = _summarize(result)
            summary["wall_seconds"] = round(wall, 2)
            entry[variant] = summary
            add = summary.get("additive_fss") or {}
            print(
                f"  {variant:>9s}: best={summary['best_expr']}  "
                f"mse={_fmt(summary['best_mse'])}  validated_mse={_fmt(summary['best_final_validated_mse'])}  "
                f"wall={wall:.1f}s  pre_mutation_early_return={add.get('pre_mutation_early_return')}"
            )
        report.append(entry)

    out_dir = Path(args.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "gs_fss_seeding_summary.json"

    def _clean(obj):
        if isinstance(obj, dict):
            return {str(k): _clean(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_clean(v) for v in obj]
        if isinstance(obj, (str, int, float, bool)) or obj is None:
            return obj
        return repr(obj)

    out.write_text(json.dumps(_clean({"cases": report}), indent=2, allow_nan=True), encoding="utf-8")
    print(f"\nSummary: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
