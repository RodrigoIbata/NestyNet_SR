# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

from __future__ import annotations

import argparse
import json
import pathlib
from typing import Any, Mapping, Sequence

from .integration import run_closed_loop_from_discovery_payload


def _load_json(path: str | pathlib.Path) -> dict[str, Any]:
    payload = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    return dict(payload) if isinstance(payload, Mapping) else {}


def _resolve_discovery_path(
    *,
    report_path: str | pathlib.Path | None = None,
    discovery_path: str | pathlib.Path | None = None,
) -> pathlib.Path:
    if discovery_path:
        return pathlib.Path(str(discovery_path))
    if not report_path:
        raise ValueError("either report_path or discovery_path is required")
    report = _load_json(report_path)
    discovery_summary = dict(report.get("discovery", {}) or {})
    summary_path = str(discovery_summary.get("results_path", "") or "").strip()
    if summary_path:
        return pathlib.Path(summary_path)
    report_obj = pathlib.Path(str(report_path))
    if report_obj.name.endswith(".report.json"):
        return report_obj.with_name(report_obj.name[: -len(".report.json")] + ".discovery.json")
    return report_obj.with_name(f"{report_obj.stem}.discovery.json")


def run_closed_loop_driver(
    *,
    report_path: str | pathlib.Path | None = None,
    discovery_path: str | pathlib.Path | None = None,
    output_path: str | pathlib.Path | None = None,
    committee_max_members: int | None = None,
    weight_temperature: float = 1.0,
    beta: float | None = None,
    gamma: float | None = None,
    disagreement_mode: str | None = None,
    lambda_cost: float | None = None,
    lambda_noise: float | None = None,
    lambda_feasibility: float | None = None,
) -> dict[str, Any]:
    resolved_discovery_path = _resolve_discovery_path(
        report_path=report_path,
        discovery_path=discovery_path,
    )
    payload = _load_json(resolved_discovery_path)
    result = run_closed_loop_from_discovery_payload(
        payload,
        committee_max_members=committee_max_members,
        weight_temperature=float(weight_temperature),
        beta=beta,
        gamma=gamma,
        disagreement_mode=disagreement_mode,
        lambda_cost=lambda_cost,
        lambda_noise=lambda_noise,
        lambda_feasibility=lambda_feasibility,
    )
    result["discovery_path"] = str(resolved_discovery_path)
    if report_path is not None:
        result["report_path_input"] = str(report_path)
    if output_path:
        out_path = pathlib.Path(str(output_path))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay the discovery closed loop from a saved SR discovery payload")
    parser.add_argument("--report", type=str, default=None, help="Optional run_SR *.report.json path")
    parser.add_argument("--discovery", type=str, default=None, help="Optional *.discovery.json path")
    parser.add_argument("--output", type=str, default=None, help="Optional output JSON path")
    parser.add_argument("--committee_max_members", type=int, default=None)
    parser.add_argument("--weight_temperature", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=None)
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--disagreement_mode", type=str, default=None, choices=["auto", "witness"])
    parser.add_argument("--lambda_cost", type=float, default=None)
    parser.add_argument("--lambda_noise", type=float, default=None)
    parser.add_argument("--lambda_feasibility", type=float, default=None)
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    result = run_closed_loop_driver(
        report_path=args.report,
        discovery_path=args.discovery,
        output_path=args.output,
        committee_max_members=None if args.committee_max_members is None else int(args.committee_max_members),
        weight_temperature=float(args.weight_temperature),
        beta=args.beta,
        gamma=args.gamma,
        disagreement_mode=args.disagreement_mode,
        lambda_cost=args.lambda_cost,
        lambda_noise=args.lambda_noise,
        lambda_feasibility=args.lambda_feasibility,
    )
    if not args.output:
        print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
