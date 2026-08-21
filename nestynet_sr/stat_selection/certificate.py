# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""Machine-readable archive-conditional Pareto certificates."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from .archive import CandidateArchive
from .audit import LossAudit
from .pareto import ConfidenceParetoResult


@dataclass(frozen=True)
class ParetoCertificate:
    """Complete declaration of what was compared and what was ruled out."""

    archive: CandidateArchive
    audit: LossAudit
    result: ConfidenceParetoResult
    schema_version: int = 2

    def __post_init__(self) -> None:
        if not self.archive.frozen:
            raise RuntimeError("candidate archive must be frozen")
        self.audit.assert_archive(self.archive)
        if self.result.archive_fingerprint not in {None, self.archive.fingerprint}:
            raise ValueError("Pareto result refers to a different candidate archive")
        if self.result.audit_fingerprint != self.audit.fingerprint:
            raise ValueError("Pareto result refers to a different loss audit")
        if tuple(self.result.candidate_ids) != tuple(self.audit.candidate_ids):
            raise ValueError("Pareto result candidate order does not match the loss audit")

    @property
    def claim(self) -> str:
        return (
            "Conditional on the frozen candidate archive, declared complexity objectives, "
            "evaluation domain, independent-unit definition, and resampling assumptions, "
            "each inferentially eligible candidate outside the practical front has a "
            "no-more-complex comparator whose risk is simultaneously superior or "
            "noninferior within the declared delta. Candidates excluded by a predeclared "
            "feasibility rule remain listed, but carry no dominance claim."
        )

    def to_dict(
        self,
        *,
        include_losses: bool = True,
        include_comparisons: bool = True,
    ) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "claim": self.claim,
            "archive": self.archive.to_dict(),
            "audit": self.audit.to_dict(include_losses=include_losses),
            "pareto": self.result.to_dict(include_comparisons=include_comparisons),
        }

    def write_json(
        self,
        path: str | Path,
        *,
        include_losses: bool = True,
        include_comparisons: bool = True,
    ) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(
                self.to_dict(
                    include_losses=include_losses,
                    include_comparisons=include_comparisons,
                ),
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        return output


def build_certificate(
    archive: CandidateArchive,
    audit: LossAudit,
    result: ConfidenceParetoResult,
) -> ParetoCertificate:
    return ParetoCertificate(archive=archive, audit=audit, result=result)
