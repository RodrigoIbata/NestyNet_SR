# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""Frozen, provenance-bearing candidate archives.

Search may be adaptive.  Statistical certification must not be.  The archive
is the membrane between those two regimes: candidates may be accumulated while
search is running, but the archive is frozen and fingerprinted before audit
losses are inspected.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Optional

from ._json import canonical_json, freeze_json, thaw_json
from .complexity import ComplexityVector, validate_complexity_collection


def candidate_id_for(
    canonical_structure: str,
    *,
    grammar_version: str = "unspecified",
    namespace: str = "nestynet-sr-candidate-v1",
    digest_chars: int = 24,
) -> str:
    """Return a deterministic identifier for one canonical symbolic structure."""

    structure = str(canonical_structure)
    grammar = str(grammar_version)
    n_chars = int(digest_chars)
    if n_chars < 12 or n_chars > 64:
        raise ValueError("digest_chars must lie in [12, 64]")
    payload = canonical_json(
        {
            "namespace": str(namespace),
            "grammar_version": grammar,
            "canonical_structure": structure,
        }
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:n_chars]


@dataclass(frozen=True)
class CandidateSpec:
    """One canonical candidate and everything needed to audit its provenance."""

    candidate_id: str
    canonical_structure: str
    complexity: ComplexityVector
    grammar_version: str = "unspecified"
    refit_recipe: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    provenance: tuple[Mapping[str, Any], ...] = ()

    def __post_init__(self) -> None:
        candidate_id = str(self.candidate_id).strip()
        if not candidate_id:
            raise ValueError("candidate_id must be non-empty")
        structure = str(self.canonical_structure).strip()
        if not structure:
            raise ValueError("canonical_structure must be non-empty")
        if not isinstance(self.complexity, ComplexityVector):
            raise TypeError("complexity must be a ComplexityVector")
        grammar = str(self.grammar_version).strip() or "unspecified"
        refit_recipe = freeze_json(dict(self.refit_recipe))
        metadata = freeze_json(dict(self.metadata))
        provenance = tuple(freeze_json(dict(item)) for item in self.provenance)
        object.__setattr__(self, "candidate_id", candidate_id)
        object.__setattr__(self, "canonical_structure", structure)
        object.__setattr__(self, "grammar_version", grammar)
        object.__setattr__(self, "refit_recipe", refit_recipe)
        object.__setattr__(self, "metadata", metadata)
        object.__setattr__(self, "provenance", provenance)

    @classmethod
    def from_structure(
        cls,
        canonical_structure: str,
        complexity: ComplexityVector,
        *,
        grammar_version: str = "unspecified",
        refit_recipe: Optional[Mapping[str, Any]] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        provenance: Optional[Sequence[Mapping[str, Any]]] = None,
        candidate_id: Optional[str] = None,
    ) -> "CandidateSpec":
        structure = str(canonical_structure)
        resolved_id = candidate_id or candidate_id_for(
            structure,
            grammar_version=str(grammar_version),
        )
        return cls(
            candidate_id=str(resolved_id),
            canonical_structure=structure,
            complexity=complexity,
            grammar_version=str(grammar_version),
            refit_recipe={} if refit_recipe is None else refit_recipe,
            metadata={} if metadata is None else metadata,
            provenance=() if provenance is None else tuple(provenance),
        )

    def with_provenance(self, record: Mapping[str, Any]) -> "CandidateSpec":
        frozen = freeze_json(dict(record))
        if any(canonical_json(item) == canonical_json(frozen) for item in self.provenance):
            return self
        return replace(self, provenance=self.provenance + (frozen,))

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "canonical_structure": self.canonical_structure,
            "grammar_version": self.grammar_version,
            "refit_recipe": thaw_json(self.refit_recipe),
            "complexity": self.complexity.as_dict(),
            "metadata": thaw_json(self.metadata),
            "provenance": thaw_json(self.provenance),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CandidateSpec":
        return cls(
            candidate_id=str(payload["candidate_id"]),
            canonical_structure=str(payload["canonical_structure"]),
            grammar_version=str(payload.get("grammar_version", "unspecified")),
            refit_recipe=payload.get("refit_recipe", {}),
            complexity=ComplexityVector.from_mapping(payload["complexity"]),
            metadata=payload.get("metadata", {}),
            provenance=tuple(payload.get("provenance", ())),
        )


class CandidateArchive:
    """Mutable candidate collector that becomes immutable at audit time."""

    schema_version = 1

    def __init__(self, *, archive_label: str = "", metadata: Optional[Mapping[str, Any]] = None):
        self.archive_label = str(archive_label)
        self._metadata = freeze_json({} if metadata is None else dict(metadata))
        self._candidates: Mapping[str, CandidateSpec] = {}
        self._frozen = False
        self._fingerprint: Optional[str] = None

    @property
    def frozen(self) -> bool:
        return bool(self._frozen)

    @property
    def metadata(self) -> Mapping[str, Any]:
        return self._metadata

    @property
    def candidate_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._candidates))

    @property
    def candidates(self) -> tuple[CandidateSpec, ...]:
        return tuple(self._candidates[candidate_id] for candidate_id in self.candidate_ids)

    def __len__(self) -> int:
        return len(self._candidates)

    def __contains__(self, candidate_id: object) -> bool:
        return str(candidate_id) in self._candidates

    def __getitem__(self, candidate_id: str) -> CandidateSpec:
        return self._candidates[str(candidate_id)]

    def _require_open(self) -> None:
        if self._frozen:
            raise RuntimeError("candidate archive is frozen")

    def add(self, candidate: CandidateSpec) -> CandidateSpec:
        """Add a candidate, merging duplicate provenance records deterministically."""

        self._require_open()
        if not isinstance(candidate, CandidateSpec):
            raise TypeError("candidate must be a CandidateSpec")
        existing = self._candidates.get(candidate.candidate_id)
        if existing is None:
            self._candidates[candidate.candidate_id] = candidate
            return candidate

        same_core = (
            existing.canonical_structure == candidate.canonical_structure
            and existing.grammar_version == candidate.grammar_version
            and existing.complexity == candidate.complexity
            and canonical_json(existing.refit_recipe) == canonical_json(candidate.refit_recipe)
            and canonical_json(existing.metadata) == canonical_json(candidate.metadata)
        )
        if not same_core:
            raise ValueError(
                f"candidate_id collision for {candidate.candidate_id!r}: core records differ"
            )
        merged = existing
        for record in candidate.provenance:
            merged = merged.with_provenance(record)
        self._candidates[candidate.candidate_id] = merged
        return merged

    def add_structure(
        self,
        canonical_structure: str,
        complexity: ComplexityVector,
        *,
        grammar_version: str = "unspecified",
        refit_recipe: Optional[Mapping[str, Any]] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        provenance: Optional[Sequence[Mapping[str, Any]]] = None,
        candidate_id: Optional[str] = None,
    ) -> CandidateSpec:
        return self.add(
            CandidateSpec.from_structure(
                canonical_structure,
                complexity,
                grammar_version=grammar_version,
                refit_recipe=refit_recipe,
                metadata=metadata,
                provenance=provenance,
                candidate_id=candidate_id,
            )
        )

    def add_provenance(self, candidate_id: str, record: Mapping[str, Any]) -> CandidateSpec:
        self._require_open()
        key = str(candidate_id)
        if key not in self._candidates:
            raise KeyError(key)
        updated = self._candidates[key].with_provenance(record)
        self._candidates[key] = updated
        return updated

    def freeze(self) -> "CandidateArchive":
        if not self._candidates:
            raise ValueError("cannot freeze an empty candidate archive")
        validate_complexity_collection([candidate.complexity for candidate in self.candidates])
        self._fingerprint = self._compute_fingerprint()
        self._candidates = MappingProxyType(dict(self._candidates))
        self._frozen = True
        return self

    @property
    def fingerprint(self) -> str:
        if not self._frozen or self._fingerprint is None:
            raise RuntimeError("freeze the candidate archive before requesting its fingerprint")
        return self._fingerprint

    def complexity_by_id(self) -> dict[str, ComplexityVector]:
        return {candidate.candidate_id: candidate.complexity for candidate in self.candidates}

    def assert_candidate_ids(self, candidate_ids: Sequence[str]) -> None:
        observed = tuple(str(item) for item in candidate_ids)
        expected = self.candidate_ids
        if len(observed) != len(set(observed)):
            raise ValueError("audit candidate_ids contain duplicates")
        if set(observed) != set(expected):
            missing = sorted(set(expected) - set(observed))
            extra = sorted(set(observed) - set(expected))
            raise ValueError(
                "audit/archive candidate mismatch; "
                f"missing={missing!r}, extra={extra!r}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "archive_label": self.archive_label,
            "metadata": thaw_json(self._metadata),
            "frozen": bool(self._frozen),
            "fingerprint": self._fingerprint,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
        }

    def write_json(self, path: str | Path) -> Path:
        if not self._frozen:
            raise RuntimeError("freeze the candidate archive before serialising it")
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return output

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CandidateArchive":
        archive = cls(
            archive_label=str(payload.get("archive_label", "")),
            metadata=payload.get("metadata", {}),
        )
        for item in payload.get("candidates", ()):
            archive.add(CandidateSpec.from_dict(item))
        if bool(payload.get("frozen", False)):
            archive.freeze()
            supplied = payload.get("fingerprint", None)
            if supplied is not None and str(supplied) != archive.fingerprint:
                raise ValueError("candidate archive fingerprint does not match its contents")
        return archive

    @classmethod
    def read_json(cls, path: str | Path) -> "CandidateArchive":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError("candidate archive JSON must contain an object")
        return cls.from_dict(payload)

    def _compute_fingerprint(self) -> str:
        payload = {
            "schema_version": int(self.schema_version),
            "archive_label": self.archive_label,
            "metadata": thaw_json(self._metadata),
            "candidates": [candidate.to_dict() for candidate in self.candidates],
        }
        return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()
