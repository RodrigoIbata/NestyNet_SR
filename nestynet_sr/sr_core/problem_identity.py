# SPDX-License-Identifier: MPL-2.0
"""Canonical problem identity across NestyNet_SR-owned data views."""

from __future__ import annotations

from pathlib import Path
import re


_STAT_VIEW_SUFFIX_RE = re.compile(
    r"\.(?:stat-search-n\d+|stat-audit-n\d+-\d+)\.[0-9a-fA-F]{8,64}$"
)


def canonical_problem_id(filepath_or_stem: object) -> str:
    """Return a CSV stem with terminal statistical-view suffixes removed."""
    raw = str(filepath_or_stem)
    name = Path(raw).name
    if name.lower().endswith(".csv"):
        name = name[:-4]
    while True:
        canonical = _STAT_VIEW_SUFFIX_RE.sub("", name)
        if canonical == name:
            return canonical
        name = canonical


__all__ = ["canonical_problem_id"]
