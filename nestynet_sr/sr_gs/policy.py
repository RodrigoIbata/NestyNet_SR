# SPDX-License-Identifier: MPL-2.0
"""Shared policy helpers for the generalized-symmetry layer."""
from __future__ import annotations

VALID_GS_POLICIES = ("augment", "replace-shadowed", "gs-only-affine")


def canonical_gs_policy(policy: str | None) -> str:
    p = str(policy or "augment").strip().lower().replace("_", "-")
    aliases = {
        "additive": "augment",
        "baseline-plus-gs": "augment",
        "baseline+gs": "augment",
        "replace": "replace-shadowed",
        "replace-baseline": "replace-shadowed",
        "replace-shadow": "replace-shadowed",
        "gs-only": "gs-only-affine",
        "affine-only": "gs-only-affine",
    }
    p = aliases.get(p, p)
    if p not in VALID_GS_POLICIES:
        return "augment"
    return p


def policy_replaces_affine_shadow(policy: str | None) -> bool:
    return canonical_gs_policy(policy) in {"replace-shadowed", "gs-only-affine"}


def policy_replaces_jet_separability(policy: str | None) -> bool:
    return canonical_gs_policy(policy) == "replace-shadowed"
