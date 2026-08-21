# SPDX-License-Identifier: MPL-2.0

"""Benchmark harness for fixed-split micro-search datasets."""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import statistics
from typing import Any, Mapping, Sequence

import torch

from .config import FactorizedSearchConfig
from .micro_search import (
    DEFAULT_MICRO_SEARCH_SPLITS,
    MicroSearchGrammar,
    _parse_constants,
    _parse_ops,
    _stable_id,
    generate_micro_search_dataset,
)


DEFAULT_POLICIES = ("inverse", "residual", "oracle", "random")


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, pathlib.Path):
        return str(value)
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, float):
        if math.isfinite(value):
            return float(value)
        return None
    return str(value)


def _safe_float(value: Any, default: float = float("inf")) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _coerce_dataset_rows(payload: Mapping[str, Any] | Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if isinstance(payload, Mapping):
        rows = [dict(row) for row in list(payload.get("rows", []) or []) if isinstance(row, Mapping)]
        meta = dict(payload)
        meta["rows"] = None
        return rows, meta
    return [dict(row) for row in list(payload or []) if isinstance(row, Mapping)], {}


def _normalize_budget_ladder(
    rows: Sequence[Mapping[str, Any]],
    *,
    payload_meta: Mapping[str, Any] | None = None,
    budget_ladder: Sequence[int] | None = None,
) -> tuple[int, ...]:
    if budget_ladder is not None:
        out = sorted({max(1, int(v)) for v in budget_ladder})
        return tuple(out or (1,))
    config = dict((payload_meta or {}).get("config", {}) or {})
    if config.get("budget_ladder", None):
        out = sorted({max(1, int(v)) for v in list(config.get("budget_ladder", []) or [])})
        if out:
            return tuple(out)
    for row in rows:
        solve_at_budget = (((row.get("metrics", {}) or {}).get("inverse", {}) or {}).get("solve_at_budget", {}) or {})
        out = sorted({max(1, int(v)) for v in list(solve_at_budget.keys())})
        if out:
            return tuple(out)
    return (1, 3, 5, 10)


def _row_solve_threshold(row: Mapping[str, Any]) -> float:
    inv = ((row.get("metrics", {}) or {}).get("inverse", {}) or {})
    return _safe_float(inv.get("solve_threshold", 1.0e-12), 1.0e-12)


def _merged_completion_rows(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    rankings = dict(row.get("rankings", {}) or {})
    for key in ("inverse", "residual"):
        for cand in list(rankings.get(key, []) or []):
            if not isinstance(cand, Mapping):
                continue
            expr = str(cand.get("expr", "") or "")
            if expr == "" or expr in seen:
                continue
            seen.add(expr)
            out.append(dict(cand))
    return out


def _rank_rows_for_policy(row: Mapping[str, Any], policy: str) -> list[dict[str, Any]]:
    rankings = dict(row.get("rankings", {}) or {})
    policy_name = str(policy).strip().lower()
    if policy_name in ("inverse", "residual"):
        out = [dict(cand) for cand in list(rankings.get(policy_name, []) or []) if isinstance(cand, Mapping)]
        out.sort(key=lambda cand: (_safe_int(cand.get("rank", 10**9), 10**9), str(cand.get("expr", ""))))
        return out

    rows = _merged_completion_rows(row)
    if policy_name == "oracle":
        rows.sort(
            key=lambda cand: (
                _safe_float(cand.get("full_probe_mse", float("inf")), float("inf")),
                _safe_float(cand.get("full_fit_mse", float("inf")), float("inf")),
                str(cand.get("expr", "")),
            )
        )
        return rows

    if policy_name == "random":
        state_id = str(row.get("state_id", ""))
        rows.sort(key=lambda cand: _stable_id("random", state_id, str(cand.get("expr", ""))))
        return rows

    raise ValueError(f"Unsupported policy {policy!r}")


def _rank_rows_for_learned_policy(row: Mapping[str, Any], learned_bundle: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    if learned_bundle is None:
        raise ValueError("Policy 'learned' requested but no learned_bundle was provided")
    from .micro_search_proposer import predict_micro_search_proposer

    return predict_micro_search_proposer(learned_bundle, row, topk=None)


def _evaluate_ranked_rows(
    ranked_rows: Sequence[Mapping[str, Any]],
    *,
    budgets: Sequence[int],
    solve_threshold: float,
) -> dict[str, Any]:
    truth_rank = None
    for idx, cand in enumerate(ranked_rows, start=1):
        if bool(cand.get("is_truth", False)):
            truth_rank = int(idx)
            break

    out: dict[str, Any] = {
        "truth_rank": truth_rank,
        "truth_at_budget": {},
        "solve_at_budget": {},
        "best_full_probe_mse_at_budget": {},
    }
    for budget in budgets:
        limit = max(1, int(budget))
        prefix = list(ranked_rows[:limit])
        truth_seen = any(bool(cand.get("is_truth", False)) for cand in prefix)
        best_prefix = None if not prefix else min(prefix, key=lambda cand: _safe_float(cand.get("full_probe_mse", float("inf")), float("inf")))
        best_mse = None if best_prefix is None else _safe_float(best_prefix.get("full_probe_mse", float("inf")), float("inf"))
        solved = bool(best_mse is not None and math.isfinite(best_mse) and best_mse <= float(solve_threshold))
        out["truth_at_budget"][limit] = bool(truth_seen)
        out["solve_at_budget"][limit] = bool(solved)
        out["best_full_probe_mse_at_budget"][limit] = None if best_mse is None or not math.isfinite(best_mse) else float(best_mse)
    return out


def _mean(values: Sequence[float]) -> float | None:
    xs = [float(v) for v in values if math.isfinite(float(v))]
    if not xs:
        return None
    return float(sum(xs) / len(xs))


def _median(values: Sequence[float]) -> float | None:
    xs = [float(v) for v in values if math.isfinite(float(v))]
    if not xs:
        return None
    return float(statistics.median(xs))


def evaluate_micro_search_dataset(
    payload: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    *,
    policies: Sequence[str] = DEFAULT_POLICIES,
    budget_ladder: Sequence[int] | None = None,
    splits: Sequence[str] = DEFAULT_MICRO_SEARCH_SPLITS,
    learned_bundle: Mapping[str, Any] | str | pathlib.Path | None = None,
) -> dict[str, Any]:
    """Evaluate deterministic baseline policies on a fixed-split micro-search dataset."""

    rows, meta = _coerce_dataset_rows(payload)
    learned_bundle_payload = learned_bundle
    if isinstance(learned_bundle_payload, (str, pathlib.Path)):
        from .micro_search_proposer import load_micro_search_proposer_bundle

        learned_bundle_payload = load_micro_search_proposer_bundle(learned_bundle_payload)
    budgets = _normalize_budget_ladder(rows, payload_meta=meta, budget_ladder=budget_ladder)
    split_names = tuple(str(split) for split in splits)
    policy_names = tuple(str(policy) for policy in policies)
    split_payload: dict[str, Any] = {}

    for split in split_names:
        split_rows = [row for row in rows if str(row.get("split", "")) == split]
        policy_payload: dict[str, Any] = {}
        for policy in policy_names:
            state_metrics: list[dict[str, Any]] = []
            for row in split_rows:
                if str(policy).strip().lower() == "learned":
                    ranked = _rank_rows_for_learned_policy(row, learned_bundle_payload)
                else:
                    ranked = _rank_rows_for_policy(row, policy)
                if not ranked:
                    continue
                metrics = _evaluate_ranked_rows(
                    ranked,
                    budgets=budgets,
                    solve_threshold=_row_solve_threshold(row),
                )
                metrics["state_id"] = str(row.get("state_id", ""))
                state_metrics.append(metrics)

            truth_ranks = [
                float(metric["truth_rank"])
                for metric in state_metrics
                if metric.get("truth_rank", None) is not None
            ]
            policy_payload[policy] = {
                "n_states": int(len(state_metrics)),
                "truth_rank_mean": _mean(truth_ranks),
                "truth_rank_median": _median(truth_ranks),
                "truth_at_budget": {
                    budget: _mean([1.0 if bool(metric["truth_at_budget"].get(budget, False)) else 0.0 for metric in state_metrics])
                    for budget in budgets
                },
                "solve_at_budget": {
                    budget: _mean([1.0 if bool(metric["solve_at_budget"].get(budget, False)) else 0.0 for metric in state_metrics])
                    for budget in budgets
                },
                "best_full_probe_mse_at_budget_mean": {
                    budget: _mean(
                        [
                            float(metric["best_full_probe_mse_at_budget"][budget])
                            for metric in state_metrics
                            if metric["best_full_probe_mse_at_budget"].get(budget, None) is not None
                        ]
                    )
                    for budget in budgets
                },
            }
        split_payload[split] = {
            "n_rows": int(len(split_rows)),
            "policies": policy_payload,
        }

    return _jsonable({
        "mode": "micro_search_benchmark",
        "n_rows": int(len(rows)),
        "budget_ladder": [int(v) for v in budgets],
        "policies": [str(policy) for policy in policy_names],
        "splits": split_payload,
        "dataset_config": dict(meta.get("config", {}) or {}),
        "dataset_split_counts": dict(meta.get("split_counts", {}) or {}),
    })


def _parse_csv_ints(raw: str) -> list[int]:
    return [int(tok.strip()) for tok in str(raw).split(",") if tok.strip()]


def _parse_csv_floats(raw: str) -> list[float]:
    return [float(tok.strip()) for tok in str(raw).split(",") if tok.strip()]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fixed-split benchmark harness for micro-search datasets")
    parser.add_argument("--dataset", default=None, help="Existing micro-search dataset JSON")
    parser.add_argument("--specs", nargs="*", default=None, help="Spec paths to generate a dataset on the fly")
    parser.add_argument("--seeds", default="0", help="Comma-separated seeds for dataset generation")
    parser.add_argument("--depth_min", type=int, default=2)
    parser.add_argument("--depth_max", type=int, default=8)
    parser.add_argument("--max_corrupt_paths_per_spec", type=int, default=None)
    parser.add_argument("--split_unit", default="spec", help="One of: spec,spec_seed,state")
    parser.add_argument("--split_fractions", default="0.7,0.15,0.15", help="Train,val,test fractions")
    parser.add_argument("--max_depth", type=int, default=2, help="Grammar max depth")
    parser.add_argument("--unary_ops", default="", help="Comma-separated unary ops")
    parser.add_argument("--binary_ops", default="add,mul", help="Comma-separated binary ops")
    parser.add_argument("--const_values", default="", help="Comma-separated literal constants")
    parser.add_argument("--n_fit", type=int, default=None)
    parser.add_argument("--n_probe", type=int, default=None)
    parser.add_argument("--poly_degree", type=int, default=None)
    parser.add_argument("--policies", default="inverse,residual,oracle,random", help="Comma-separated policies")
    parser.add_argument("--learned_bundle", default=None, help="Optional trained micro-search proposer bundle")
    parser.add_argument("--dataset_out", default=None, help="Optional path to save the generated dataset")
    parser.add_argument("--output", default=None, help="Optional path to save benchmark JSON")
    parser.add_argument("--no_enforce_dims", action="store_true")
    parser.add_argument("--no_samples", action="store_true")
    parser.add_argument("--seed", type=int, default=0, help="Reserved for future learned policies")
    return parser


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    parser = build_arg_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.dataset:
        payload = json.loads(pathlib.Path(args.dataset).read_text(encoding="utf-8"))
    else:
        if not args.specs:
            raise ValueError("Either --dataset or at least one --specs entry is required")
        hp = FactorizedSearchConfig()
        if args.n_fit is not None:
            hp.n_fit = int(args.n_fit)
        if args.n_probe is not None:
            hp.n_probe = int(args.n_probe)
        if args.poly_degree is not None:
            hp.poly_degree = int(args.poly_degree)
        grammar = MicroSearchGrammar(
            max_depth=int(args.max_depth),
            unary_ops=tuple(_parse_ops(args.unary_ops)),
            binary_ops=tuple(_parse_ops(args.binary_ops)),
            constant_values=tuple(_parse_constants(args.const_values)),
        )
        payload = generate_micro_search_dataset(
            args.specs,
            factorized_search_hp=hp,
            seeds=_parse_csv_ints(args.seeds),
            dtype=torch.float64,
            enforce_dims=not bool(args.no_enforce_dims),
            depth_min=int(args.depth_min),
            depth_max=int(args.depth_max),
            max_corrupt_paths_per_spec=args.max_corrupt_paths_per_spec,
            grammar=grammar,
            split_unit=str(args.split_unit),
            split_fractions=_parse_csv_floats(args.split_fractions),
            include_samples=not bool(args.no_samples),
            include_completion_tables=True,
            verbose=False,
        )
        if args.dataset_out:
            dataset_out = pathlib.Path(args.dataset_out)
            dataset_out.parent.mkdir(parents=True, exist_ok=True)
            dataset_out.write_text(json.dumps(_jsonable(payload), indent=2), encoding="utf-8")

    benchmark = evaluate_micro_search_dataset(
        payload,
        policies=[tok.strip() for tok in str(args.policies).split(",") if tok.strip()],
        learned_bundle=args.learned_bundle,
    )
    text = json.dumps(_jsonable(benchmark), indent=2)
    if args.output:
        out_path = pathlib.Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text, encoding="utf-8")
    else:
        print(text)
    return benchmark


if __name__ == "__main__":  # pragma: no cover
    main()
