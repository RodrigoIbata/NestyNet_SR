# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from ..atom_policy import atom_policy_family_scores
from ..basis_state import BasisState, ProposalContext
from .scaffold_enum import normalize_families


@dataclass(frozen=True)
class FamilyBudgetPlanEntry:
    family: str
    max_scaffolds: int
    anchors_per_family: int
    priority_score: float
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "family": str(self.family),
            "max_scaffolds": int(self.max_scaffolds),
            "anchors_per_family": int(self.anchors_per_family),
            "priority_score": float(self.priority_score),
            "reason": str(self.reason),
        }


def _snapshot_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, Mapping):
        parts: list[str] = []
        for key, item in dict(value).items():
            parts.append(str(key))
            parts.append(_snapshot_text(item))
        return " ".join(parts)
    if isinstance(value, (list, tuple)):
        return " ".join(_snapshot_text(item) for item in value)
    return str(value)


def _context_basis_state(context: ProposalContext | None) -> BasisState | None:
    if isinstance(context, ProposalContext) and isinstance(context.basis_state, BasisState):
        return context.basis_state
    return None


def heuristic_family_priority_scores(
    *,
    families: Sequence[str] | None,
    context: ProposalContext | None = None,
) -> dict[str, float]:
    scores, _decomposition = heuristic_family_priority_scores_with_decomposition(
        families=families,
        context=context,
    )
    return scores


def heuristic_family_priority_scores_with_decomposition(
    *,
    families: Sequence[str] | None,
    context: ProposalContext | None = None,
) -> tuple[dict[str, float], dict[str, Any]]:
    families_norm = normalize_families(families)
    if not families_norm:
        return {}, {}

    scores = {family: 0.0 for family in families_norm}
    decomposition: dict[str, dict[str, Any]] = {
        family: {
            "legacy": 0.0,
            "residual_probe": 0.0,
            "atom_policy": 0.0,
            "basis_family": 0.0,
            "top_atom_reasons": [],
        }
        for family in families_norm
    }
    hints = dict(getattr(context, "family_hints", {}) or {}) if isinstance(context, ProposalContext) else {}
    for family, raw_value in list(hints.items()):
        token = str(family or "").strip().lower()
        if token not in scores:
            continue
        try:
            value = float(raw_value)
        except Exception:
            continue
        scores[token] += value
        decomposition[token]["residual_probe"] += value

    text = ""
    if isinstance(context, ProposalContext):
        text = " ".join(
            part
            for part in (
                _snapshot_text(context.residual_witness),
                _snapshot_text(context.diagnostics),
            )
            if part
        ).lower()
    if text:
        if any(token in text for token in ("periodic", "sin", "cos", "oscill", "harmonic")):
            scores.setdefault("periodic", 0.0)
            if "periodic" in decomposition:
                decomposition["periodic"]["legacy"] += 2.0
                scores["periodic"] += 2.0
        if any(token in text for token in ("exp", "exponential", "growth")):
            scores.setdefault("exp", 0.0)
            if "exp" in decomposition:
                decomposition["exp"]["legacy"] += 2.0
                scores["exp"] += 2.0
        if any(token in text for token in ("log", "logarith")):
            scores.setdefault("log", 0.0)
            if "log" in decomposition:
                decomposition["log"]["legacy"] += 2.0
                scores["log"] += 2.0
        if any(token in text for token in ("rational", "quotient", "pole", "division", "ratio", "reciprocal", "denominator")):
            scores.setdefault("rational", 0.0)
            if "rational" in decomposition:
                decomposition["rational"]["legacy"] += 2.0
                scores["rational"] += 2.0
        if any(token in text for token in ("power", "sqrt", "root", "inverse_power", "invsqrt")):
            scores.setdefault("power", 0.0)
            if "power" in decomposition:
                decomposition["power"]["legacy"] += 2.0
                scores["power"] += 2.0
        if any(token in text for token in ("quadratic", "radial", "norm", "sumsq")):
            scores.setdefault("quadratic", 0.0)
            if "quadratic" in decomposition:
                decomposition["quadratic"]["legacy"] += 2.0
                scores["quadratic"] += 2.0

    basis_state = _context_basis_state(context)
    if basis_state is not None:
        existing = [str(block.family or "").strip().lower() for block in basis_state.blocks]
        for family in families_norm:
            if family in existing:
                scores[family] += 0.25
                decomposition[family]["basis_family"] += 0.25

    atom_scores, atom_decomposition = atom_policy_family_scores(
        getattr(context, "atom_library", None) if isinstance(context, ProposalContext) else None,
        families_norm,
    )
    for family, value in atom_scores.items():
        if family not in scores:
            continue
        bounded = min(3.0, max(0.0, float(value)))
        scores[family] += bounded
        decomposition[family]["atom_policy"] += bounded
        reasons = dict(atom_decomposition.get(family, {}) or {}).get("top_atom_reasons", [])
        decomposition[family]["top_atom_reasons"] = list(reasons or [])[:5]

    return (
        {family: float(scores.get(family, 0.0)) for family in families_norm},
        {family: dict(decomposition.get(family, {})) for family in families_norm},
    )


def allocate_family_budgets(
    *,
    families: Sequence[str] | None,
    max_scaffolds: int,
    anchors_per_family: int,
    context: ProposalContext | None = None,
) -> dict[str, Any]:
    families_norm = normalize_families(families)
    max_scaffolds_i = max(0, int(max_scaffolds))
    anchor_cap = max(0, int(anchors_per_family))
    if not families_norm or max_scaffolds_i <= 0:
        return {
            "steered": False,
            "scores": {},
            "entries": [],
        }

    scores, decomposition = heuristic_family_priority_scores_with_decomposition(
        families=families_norm,
        context=context,
    )
    has_signal = any(abs(float(score)) > 1.0e-12 for score in scores.values())
    if not has_signal:
        # No steering signal — give each family a fair share of the budget
        # so that no single family can consume everything.
        n_fam = max(1, len(families_norm))
        fair_share = int(max_scaffolds_i // n_fam)
        remainder = int(max_scaffolds_i % n_fam)
        entries = []
        for idx, family in enumerate(families_norm):
            budget = int(fair_share + (1 if idx < remainder else 0))
            if budget <= 0:
                continue
            entries.append(
                FamilyBudgetPlanEntry(
                    family=family,
                    max_scaffolds=budget,
                    anchors_per_family=anchor_cap,
                    priority_score=float(scores.get(family, 0.0)),
                    reason="legacy_order",
                ).to_dict()
            )
        return {
            "steered": False,
            "scores": scores,
            "score_decomposition": decomposition,
            "entries": entries,
        }

    ranked = sorted(
        enumerate(families_norm),
        key=lambda item: (-float(scores.get(item[1], 0.0)), int(item[0]), str(item[1])),
    )
    if max_scaffolds_i < len(ranked):
        ranked = ranked[:max_scaffolds_i]

    remaining = int(max_scaffolds_i)
    selected = [family for _idx, family in ranked]
    # Every family gets a minimum floor so that no operator type is starved
    # even when steering strongly favours a few families.
    min_floor = 2 if max_scaffolds_i >= 2 * max(1, len(selected)) else 1
    budgets = {family: min(min_floor, remaining) for family in selected}
    remaining -= sum(budgets.values())

    if remaining > 0 and selected:
        weights = {
            family: max(0.0, float(scores.get(family, 0.0))) + 1.0
            for family in selected
        }
        total_weight = sum(weights.values())
        if total_weight <= 0.0:
            total_weight = float(len(selected))
        for family in selected:
            if remaining <= 0:
                break
            share = int((float(weights[family]) / float(total_weight)) * float(remaining))
            if share <= 0:
                continue
            budgets[family] += int(share)
        used = sum(int(v) for v in budgets.values())
        leftover = max(0, int(max_scaffolds_i) - int(used))
        idx = 0
        while leftover > 0 and selected:
            family = selected[idx % len(selected)]
            budgets[family] += 1
            leftover -= 1
            idx += 1

    entries = [
        FamilyBudgetPlanEntry(
            family=family,
            max_scaffolds=int(budgets.get(family, 1)),
            anchors_per_family=anchor_cap,
            priority_score=float(scores.get(family, 0.0)),
            reason="heuristic_priority",
        ).to_dict()
        for family in selected
    ]
    return {
        "steered": True,
        "scores": scores,
        "score_decomposition": decomposition,
        "entries": entries,
    }


__all__ = [
    "FamilyBudgetPlanEntry",
    "allocate_family_budgets",
    "heuristic_family_priority_scores",
    "heuristic_family_priority_scores_with_decomposition",
]
