#!/usr/bin/env python3
# SPDX-License-Identifier: MPL-2.0

"""Emit a matched GS ablation plan.

This is a lightweight Phase-5 scaffold. It does not run benchmarks; it writes a
JSON manifest and shell command list for the six-arm design needed to separate
vocabulary expansion from symmetry evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import sys
from pathlib import Path


def _split(text: str) -> list[str]:
    return shlex.split(str(text or ""))


def _quote(cmd: list[str]) -> str:
    return " ".join(shlex.quote(str(x)) for x in cmd)


def _arm_command(base: list[str], output_flag: str, out_dir: Path, extra: list[str]) -> list[str]:
    cmd = list(base)
    if output_flag and output_flag not in cmd:
        cmd += [output_flag, str(out_dir)]
    cmd += list(extra)
    return cmd


def _flag_value(cmd: list[str], flag: str, default: str | None = None) -> str | None:
    for i, tok in enumerate(cmd):
        if tok == flag and i + 1 < len(cmd):
            return str(cmd[i + 1])
    return default


def _has_flag(cmd: list[str], flag: str) -> bool:
    return flag in set(cmd)


def _mechanism_summary(cmd: list[str]) -> dict:
    mode = str(_flag_value(cmd, "--gs-mode", "propose") or "propose").lower()
    gs_enabled = _has_flag(cmd, "--gs-enable") and mode != "off"
    hard_tail = _has_flag(cmd, "--de-hard-tail-templates") or _has_flag(cmd, "--gs-de-templates")
    hard_tail_velocity = _has_flag(cmd, "--de-hard-tail-velocity-templates") or _has_flag(cmd, "--gs-de-velocity-templates")
    hard_tail_radial = not (
        _has_flag(cmd, "--de-hard-tail-no-radial-templates")
        or _has_flag(cmd, "--gs-de-no-radial-templates")
    )
    lie_prolongation = gs_enabled and _has_flag(cmd, "--gs-de-lie-prolongation")
    lie_selection = gs_enabled and _has_flag(cmd, "--gs-de-lie-use-for-selection")
    summary = {
        "gs_enabled": bool(gs_enabled),
        "gs_mode": mode if gs_enabled else "off",
        "gs_proposal_mode": bool(gs_enabled and mode in {"propose", "auto"}),
        "heldout_verified_proposals": False,
        "neutral_hard_tail_priors": bool(hard_tail),
        "neutral_hard_tail_velocity_priors": bool(hard_tail_velocity),
        "neutral_hard_tail_radial_priors": bool(hard_tail and hard_tail_radial),
        "lie_prolongation_audit": bool(lie_prolongation),
        "lie_prolongation_selection": bool(lie_selection and lie_prolongation),
    }
    summary["mechanism_hash"] = hashlib.sha256(
        json.dumps(summary, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    return summary


def _annotate_arm(arm: dict) -> dict:
    arm = dict(arm)
    arm["mechanisms"] = _mechanism_summary(list(arm["command"]))
    return arm


def _validate_plan(arms: list[dict], *, allow_collapsed_arms: bool = False) -> list[dict]:
    by_id = {str(arm["id"]): arm for arm in arms}
    errors: list[str] = []
    warnings: list[str] = []
    if not by_id["B"]["mechanisms"]["neutral_hard_tail_priors"]:
        errors.append("Arm B must enable neutral hard-tail vocabulary.")
    if by_id["C"]["mechanisms"]["lie_prolongation_selection"]:
        errors.append("Arm C must not enable selection-time Lie scoring.")
    if by_id["D"]["mechanisms"]["heldout_verified_proposals"]:
        errors.append("Arm D default planner should not claim held-out verification.")
    if by_id["E"]["mechanisms"]["lie_prolongation_selection"] and not by_id["E"]["mechanisms"]["lie_prolongation_audit"]:
        errors.append("Arm E selection requires active Lie prolongation audit.")
    if _has_flag(by_id["E"]["command"], "--gs-de-lie-use-for-selection") and not _has_flag(
        by_id["E"]["command"], "--gs-de-lie-prolongation"
    ):
        errors.append("Arm E must pass --gs-de-lie-prolongation with --gs-de-lie-use-for-selection.")
    hashes = {aid: by_id[aid]["mechanisms"]["mechanism_hash"] for aid in ("C", "D", "E")}
    if len(set(hashes.values())) != len(hashes):
        msg = f"C/D/E mechanism hashes collapsed: {hashes}"
        if allow_collapsed_arms:
            warnings.append(msg)
        else:
            errors.append(msg)
    if errors:
        raise ValueError("; ".join(errors))
    return [{"level": "warning", "message": w} for w in warnings]


def build_plan(args: argparse.Namespace) -> dict:
    base = _split(args.cmd)
    if not base:
        raise ValueError("--cmd is required")
    out_root = Path(args.output_root).resolve()
    neutral = _split(args.neutral_library_args)
    gs_audit = _split(args.gs_audit_args)
    verified = _split(args.verified_gs_args)
    gs_selection = _split(args.gs_selection_args)
    oracle = _split(args.oracle_args)

    arms = [
        {
            "id": "A",
            "name": "baseline",
            "purpose": "Original baseline without GS or added hard-tail vocabulary.",
            "command": _arm_command(base, args.output_flag, out_root / "A_baseline", []),
        },
        {
            "id": "B",
            "name": "neutral_vocabulary",
            "purpose": "Baseline plus the same hard-tail vocabulary under neutral labels.",
            "command": _arm_command(base, args.output_flag, out_root / "B_neutral_vocabulary", neutral),
        },
        {
            "id": "C",
            "name": "gs_audit_only",
            "purpose": "GS diagnostics/reporting only; no proposal or selection effect.",
            "command": _arm_command(base, args.output_flag, out_root / "C_gs_audit_only", gs_audit),
        },
        {
            "id": "D",
            "name": "gs_proposal_mode_unverified",
            "purpose": "GS proposal-mode diagnostics/prototype hooks without held-out verification; not a headline proposal arm.",
            "command": _arm_command(base, args.output_flag, out_root / "D_gs_proposal_mode_unverified", verified),
        },
        {
            "id": "E",
            "name": "lie_audit_plus_explicit_selection",
            "purpose": "Explicitly enabled Lie-prolongation audit with selection weight; exploratory only.",
            "command": _arm_command(
                base,
                args.output_flag,
                out_root / "E_lie_audit_selection",
                verified + gs_selection,
            ),
        },
        {
            "id": "F",
            "name": "oracle_template_upper_bound",
            "purpose": "Oracle or target-specific templates for debugging only, not a GS headline.",
            "command": _arm_command(base, args.output_flag, out_root / "F_oracle_template_upper_bound", oracle),
        },
    ]
    arms = [_annotate_arm(arm) for arm in arms]
    validation = _validate_plan(
        arms,
        allow_collapsed_arms=bool(getattr(args, "allow_collapsed_arms", False)),
    )
    return {
        "schema": "nestynet_sr_gs_matched_ablation_plan_v1",
        "description": "Matched search-budget design for separating GS evidence from vocabulary expansion.",
        "key_comparisons": [
            "A_vs_B isolates neutral hard-tail vocabulary expansion.",
            "C is diagnostics-only and should match A in search behavior.",
            "D is unverified GS proposal-mode plumbing and must not be used as held-out evidence.",
            "D_vs_E isolates explicit Lie-prolongation selection only when E has an active scorer.",
            "F is an upper bound for debugging and must not be used as a GS headline.",
        ],
        "validation": validation,
        "arms": arms,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--cmd", required=True, help="Base benchmark command, quoted as one shell string.")
    p.add_argument("--output-root", default="results/gs_matched_ablation")
    p.add_argument("--output-flag", default="--results_dir", help="Output flag injected into each arm; use empty string to disable.")
    p.add_argument("--neutral-library-args", default="--de-hard-tail-templates --de-hard-tail-velocity-templates", help="Args for neutral hard-tail vocabulary arm.")
    p.add_argument("--gs-audit-args", default="--gs-enable --gs-mode audit")
    p.add_argument("--verified-gs-args", default="--gs-enable --gs-mode propose")
    p.add_argument("--gs-selection-args", default="--gs-de-lie-prolongation --gs-de-lie-use-for-selection")
    p.add_argument("--oracle-args", default="")
    p.add_argument("--allow-collapsed-arms", action="store_true", help="Emit warnings instead of failing when C/D/E normalize to identical mechanisms.")
    p.add_argument("--json-out", default=None)
    p.add_argument("--commands-out", default=None)
    args = p.parse_args(argv)

    plan = build_plan(args)
    if args.json_out:
        path = Path(args.json_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    if args.commands_out:
        path = Path(args.commands_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(_quote(arm["command"]) for arm in plan["arms"]) + "\n", encoding="utf-8")
    print(json.dumps(plan, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
