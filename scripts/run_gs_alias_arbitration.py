#!/usr/bin/env python3
# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata

"""Symmetry-certificate arbitration on the compositional alias block (900-903).

The dictionary-isolation audit (run_feynman_de_stlsq_dictionary_baselines.py)
showed that the standard STLSQ library can PASS rollout validation on cases
902/903 with a *structural alias* (a law that mimics the truth on the sampled
range).  This experiment asks whether symmetry evidence separates the alias
from the true law without any new data:

1.  Every selected candidate (per dictionary preset) receives a determining
    certificate: which affine point symmetries does the candidate equation
    admit (on-shell recovery + off-shell relative invariance, including
    nullspace combinations)?
2.  Every certified generator is then tested against the trajectory ensemble
    alone: an equation symmetry maps solutions to other solutions, so
    ``exp(eps*V)`` must map one measured trajectory onto another.
3.  Arbitration: a candidate is penalized for every certified generator the
    data refute, and rewarded for every certified generator the data support.

The truth is never consulted for arbitration; it is only used to score the
final verdicts.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from run_feynman_de_stlsq_dictionary_baselines import (  # noqa: E402
    DATA_DIR,
    BENCHMARK_FILE,
    load_existing_runs,
    load_problems,
    _load_xy,
)

from nestynet_sr.sr_gs.de_certificates import (  # noqa: E402
    certify_scalar_ode_candidate,
    generator_ensemble_support,
)

TRUE_PRESET = "oracle"
# Deliberate structural-alias presets from the audit: the standard library
# lacks the required atoms; expanded_unary lacks the carrier products.
ALIAS_PRESETS = ("standard", "expanded_unary")


def _load_summary(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _case_trajectories(problem: Any, data_dir: Path, n_traj: int) -> list[tuple[np.ndarray, np.ndarray]]:
    runs, _source = load_existing_runs(problem, data_dir, n_traj=n_traj)
    out = []
    for run in runs:
        x, u = _load_xy(run)
        out.append((np.asarray(x, dtype=np.float64), np.asarray(u, dtype=np.float64)))
    return out


def certify_candidate_row(
    row: dict[str, Any],
    trajectories: list[tuple[np.ndarray, np.ndarray]],
    *,
    cert_tol: float,
    coeff_prune_tol: float,
    seed: int,
) -> dict[str, Any]:
    names = list(row.get("selected_term_names", []))
    coeffs = [float(v) for v in row.get("selected_coefficients", [])]
    if not names:
        return {"status": "empty_candidate"}
    x_all = np.concatenate([t[0] for t in trajectories])
    u_all = np.concatenate([t[1] for t in trajectories])
    result = certify_scalar_ode_candidate(
        x=x_all,
        u=u_all,
        coeffs=coeffs,
        term_names=names,
        order=1,
        coeff_prune_tol=float(coeff_prune_tol),
        on_shell_tol=float(cert_tol),
        off_shell_tol=float(cert_tol),
        seed=int(seed),
    )
    return result.to_report()


def data_support_for_generators(
    generator_rows: list[dict[str, Any]],
    trajectories: list[tuple[np.ndarray, np.ndarray]],
    *,
    flow_rel_tol: float,
    cache: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    out = []
    for gen in generator_rows:
        key = json.dumps([round(float(v), 10) for v in gen["coefficients"]])
        if key not in cache:
            cache[key] = generator_ensemble_support(
                trajectories,
                gen["coefficients"],
                support_rel_tol=float(flow_rel_tol),
            )
        support = cache[key]
        out.append(
            {
                "name": gen["name"],
                "family": gen["family"],
                "coefficients": gen["coefficients"],
                "multiplier": gen["multiplier"],
                "flow_status": support.get("status"),
                "data_supported": support.get("supported"),
                "median_best_rel_rms": support.get("median_best_rel_rms"),
                "fraction_pairs_supporting": support.get("fraction_pairs_supporting"),
            }
        )
    return out


def arbitration_score(gen_rows: list[dict[str, Any]]) -> tuple[int, int, int]:
    supported = sum(1 for g in gen_rows if g.get("data_supported") is True)
    refuted = sum(1 for g in gen_rows if g.get("data_supported") is False)
    return supported - refuted, supported, refuted


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Symmetry-certificate arbitration for the 900-903 alias block",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--baselines_json",
        default=str(REPO_ROOT / "results" / "feynman_de_stlsq_dictionary_baselines" /
                    "stlsq_dictionary_baselines_summary.json"),
    )
    parser.add_argument("--benchmark_file", default=str(BENCHMARK_FILE))
    parser.add_argument("--data_dir", default=str(DATA_DIR))
    parser.add_argument("--ids", default="900,901,902,903")
    parser.add_argument("--n_traj", type=int, default=6)
    parser.add_argument("--cert_tol", type=float, default=3.0e-3,
                        help="certificate tolerance; must absorb the fitted-coefficient noise "
                        "of STLSQ candidates (small spurious offsets)")
    parser.add_argument("--coeff_prune_tol", type=float, default=0.0)
    parser.add_argument("--flow_rel_tol", type=float, default=5.0e-3,
                        help="relative RMS below which a flowed trajectory pair counts as a match")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--results_dir", default=str(REPO_ROOT / "results" / "gs_alias_arbitration"))
    args = parser.parse_args()

    summary = _load_summary(Path(args.baselines_json))
    rows_by_case: dict[str, dict[str, dict[str, Any]]] = {}
    for row in summary["rows"]:
        rows_by_case.setdefault(str(row["id"]), {})[str(row["preset"])] = row

    problems = load_problems(args.benchmark_file)
    ids = [s.strip() for s in str(args.ids).split(",") if s.strip()]
    data_dir = Path(args.data_dir).resolve()

    report_cases = []
    for pid in ids:
        problem = problems[pid]
        trajectories = _case_trajectories(problem, data_dir, int(args.n_traj))
        flow_cache: dict[str, dict[str, Any]] = {}
        case_entry: dict[str, Any] = {
            "id": pid,
            "description": str(problem.description),
            "n_traj": len(trajectories),
            "candidates": [],
        }
        print(f"\n=== de{pid}: {problem.description} ===")
        for preset in (*ALIAS_PRESETS, "expanded_closure", TRUE_PRESET):
            row = rows_by_case.get(pid, {}).get(preset)
            if row is None:
                continue
            sel = dict(row.get("selected", {}) or {})
            cert = certify_candidate_row(
                sel,
                trajectories,
                cert_tol=float(args.cert_tol),
                coeff_prune_tol=float(args.coeff_prune_tol),
                seed=int(args.seed),
            )
            gen_rows = data_support_for_generators(
                list(cert.get("generators", [])),
                trajectories,
                flow_rel_tol=float(args.flow_rel_tol),
                cache=flow_cache,
            )
            score, supported, refuted = arbitration_score(gen_rows)
            entry = {
                "preset": preset,
                "role": "truth" if preset == TRUE_PRESET else (
                    "alias" if preset in ALIAS_PRESETS else "closure"
                ),
                "rollout_status": sel.get("status"),
                "canonical_equation": sel.get("canonical_equation"),
                "required_atom_selected": sel.get("required_atom_selected"),
                "certificate_status": cert.get("status"),
                "determining_nullity": cert.get("determining_nullity"),
                "certified_generators": gen_rows,
                "arbitration": {
                    "score": score,
                    "generators_supported": supported,
                    "generators_refuted": refuted,
                },
            }
            case_entry["candidates"].append(entry)
            gen_txt = ", ".join(
                f"{g['name']}[{'DATA+' if g['data_supported'] else 'DATA-' if g['data_supported'] is False else '?'}]"
                for g in gen_rows
            ) or "(none)"
            print(
                f"  {preset:>16s} [{entry['role']:>7s}] rollout={str(sel.get('status')):>7s} "
                f"score={score:+d} (+{supported}/-{refuted})  gens: {gen_txt}"
            )
            print(f"                    eq: {sel.get('canonical_equation')}")

        # arbitration verdict: highest score wins; ties broken by rollout rank
        status_rank = {"PASS": 0, "PARTIAL": 1, "FAIL": 2, "ERROR": 3, None: 4}
        scored = [
            (c["arbitration"]["score"], -status_rank.get(c["rollout_status"], 4), c["preset"])
            for c in case_entry["candidates"]
        ]
        winner = max(scored)[2] if scored else None
        case_entry["arbitration_winner_preset"] = winner
        winner_role = next(
            (c["role"] for c in case_entry["candidates"] if c["preset"] == winner), None
        )
        case_entry["arbitration_winner_role"] = winner_role
        case_entry["winner_is_true_structure"] = bool(winner_role in ("truth", "closure"))
        print(f"  --> arbitration winner: {winner} ({winner_role})")
        report_cases.append(case_entry)

    payload = {
        "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "note": (
            "Certificates are intrinsic to each candidate equation; the trajectory ensemble "
            "is consulted only through the exp(eps*V) flow test. The oracle/closure rows serve "
            "as the true-structure reference; arbitration never sees the truth labels."
        ),
        "cases": report_cases,
    }
    out_dir = Path(args.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "gs_alias_arbitration_summary.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nSummary: {out_path}")
    n_correct = sum(1 for c in report_cases if c["winner_is_true_structure"])
    print(f"Arbitration selected the true structure in {n_correct}/{len(report_cases)} cases")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
