# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""Common-domain, independent-unit loss audits.

Rows in this table are the declared independent statistical units, not
necessarily samples in a tensor.  For DE discovery a row will normally be a
whole trajectory, experiment, excitation, or field realisation.  Columns are
frozen candidates.  Every candidate is evaluated on every row; domain and
integration failures are recorded rather than silently dropped.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import json
import math
from types import MappingProxyType
from typing import Any, Optional

import numpy as np

from ._json import canonical_json, freeze_mapping, thaw_json
from .archive import CandidateArchive


def _readonly_array(value: Any, *, dtype: Any, ndim: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=dtype)
    if array.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions, got shape {array.shape!r}")
    out = np.array(array, dtype=dtype, copy=True)
    out.setflags(write=False)
    return out


def _normalised_weights(n_units: int, weights: Optional[Sequence[float]]) -> np.ndarray:
    if weights is None:
        out = np.full(int(n_units), 1.0 / float(n_units), dtype=np.float64)
    else:
        out = np.asarray(weights, dtype=np.float64)
        if out.shape != (int(n_units),):
            raise ValueError(
                f"unit_weights must have shape {(int(n_units),)!r}, got {out.shape!r}"
            )
        if not np.all(np.isfinite(out)):
            raise ValueError("unit_weights must be finite")
        if np.any(out <= 0.0):
            raise ValueError("unit_weights must be strictly positive")
        total = float(np.sum(out))
        if not math.isfinite(total) or total <= 0.0:
            raise ValueError("unit_weights must have a positive finite sum")
        out = out / total
    out = np.array(out, dtype=np.float64, copy=True)
    out.setflags(write=False)
    return out


@dataclass(frozen=True)
class AuditDesign:
    """Predeclared statistical design for one candidate audit."""

    loss_name: str
    unit_kind: str
    fit_protocol: str
    evaluation_domain: Mapping[str, Any] = field(default_factory=dict)
    sampling_assumptions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        loss_name = str(self.loss_name).strip()
        unit_kind = str(self.unit_kind).strip()
        fit_protocol = str(self.fit_protocol).strip()
        if not loss_name:
            raise ValueError("loss_name must be non-empty")
        if not unit_kind:
            raise ValueError("unit_kind must be non-empty")
        if not fit_protocol:
            raise ValueError("fit_protocol must be non-empty")
        assumptions = tuple(str(item).strip() for item in self.sampling_assumptions)
        if any(not item for item in assumptions):
            raise ValueError("sampling assumptions must be non-empty strings")
        object.__setattr__(self, "loss_name", loss_name)
        object.__setattr__(self, "unit_kind", unit_kind)
        domain = freeze_mapping(self.evaluation_domain)
        if not domain:
            raise ValueError("evaluation_domain must contain an explicit domain declaration")
        object.__setattr__(self, "fit_protocol", fit_protocol)
        object.__setattr__(self, "evaluation_domain", domain)
        object.__setattr__(self, "sampling_assumptions", assumptions)

    def to_dict(self) -> dict[str, Any]:
        return {
            "loss_name": self.loss_name,
            "unit_kind": self.unit_kind,
            "fit_protocol": self.fit_protocol,
            "evaluation_domain": thaw_json(self.evaluation_domain),
            "sampling_assumptions": list(self.sampling_assumptions),
        }


@dataclass(frozen=True)
class UnitLossRecord:
    """Losses for every frozen candidate on one independent evaluation unit."""

    unit_id: str
    losses: Mapping[str, float]
    failures: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        unit_id = str(self.unit_id).strip()
        if not unit_id:
            raise ValueError("unit_id must be non-empty")
        losses = {str(key): float(value) for key, value in self.losses.items()}
        failures = tuple(sorted({str(item) for item in self.failures}))
        object.__setattr__(self, "unit_id", unit_id)
        object.__setattr__(self, "losses", MappingProxyType(losses))
        object.__setattr__(self, "failures", failures)
        object.__setattr__(self, "metadata", freeze_mapping(self.metadata))


@dataclass(frozen=True)
class LossAudit:
    """Paired candidate losses on a declared common set of independent units."""

    candidate_ids: tuple[str, ...]
    unit_ids: tuple[str, ...]
    design: AuditDesign
    losses: np.ndarray
    unit_weights: np.ndarray
    failure_mask: np.ndarray
    failure_loss: Optional[float] = None
    archive_fingerprint: Optional[str] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    unit_metadata: tuple[Mapping[str, Any], ...] = ()

    def __post_init__(self) -> None:
        candidate_ids = tuple(str(item) for item in self.candidate_ids)
        unit_ids = tuple(str(item) for item in self.unit_ids)
        if not isinstance(self.design, AuditDesign):
            raise TypeError("design must be an AuditDesign")
        if not candidate_ids:
            raise ValueError("at least one candidate is required")
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("candidate_ids must be unique")
        if len(unit_ids) < 2:
            raise ValueError("at least two independent evaluation units are required")
        if len(unit_ids) != len(set(unit_ids)):
            raise ValueError("unit_ids must be unique")

        losses = _readonly_array(self.losses, dtype=np.float64, ndim=2, name="losses")
        expected_shape = (len(unit_ids), len(candidate_ids))
        if losses.shape != expected_shape:
            raise ValueError(f"losses must have shape {expected_shape!r}, got {losses.shape!r}")
        if not np.all(np.isfinite(losses)):
            raise ValueError(
                "LossAudit losses must be finite; use from_matrix(..., nonfinite='penalize') "
                "to retain failed candidates on the common domain"
            )

        failures = _readonly_array(
            self.failure_mask,
            dtype=np.bool_,
            ndim=2,
            name="failure_mask",
        )
        if failures.shape != expected_shape:
            raise ValueError(
                f"failure_mask must have shape {expected_shape!r}, got {failures.shape!r}"
            )

        weights = _normalised_weights(len(unit_ids), self.unit_weights)
        if float(np.sum(weights * weights)) >= 1.0:
            raise ValueError("unit weights do not leave more than one effective unit")

        failure_loss = self.failure_loss
        if failure_loss is not None:
            failure_loss = float(failure_loss)
            if not math.isfinite(failure_loss):
                raise ValueError("failure_loss must be finite")
        if np.any(failures):
            if failure_loss is None:
                raise ValueError("failure_mask requires a declared finite failure_loss")
            if not np.allclose(
                losses[failures],
                failure_loss,
                rtol=0.0,
                atol=32.0 * np.finfo(np.float64).eps * max(1.0, abs(failure_loss)),
            ):
                raise ValueError("all failed candidate/unit cells must equal failure_loss")

        unit_metadata = tuple(self.unit_metadata)
        if not unit_metadata:
            unit_metadata = tuple(MappingProxyType({}) for _ in unit_ids)
        if len(unit_metadata) != len(unit_ids):
            raise ValueError("unit_metadata must contain one record per unit")
        unit_metadata = tuple(freeze_mapping(item) for item in unit_metadata)

        object.__setattr__(self, "candidate_ids", candidate_ids)
        object.__setattr__(self, "unit_ids", unit_ids)
        object.__setattr__(self, "losses", losses)
        object.__setattr__(self, "unit_weights", weights)
        object.__setattr__(self, "failure_mask", failures)
        object.__setattr__(self, "failure_loss", failure_loss)
        object.__setattr__(self, "archive_fingerprint", None if self.archive_fingerprint is None else str(self.archive_fingerprint))
        object.__setattr__(self, "metadata", freeze_mapping(self.metadata))
        object.__setattr__(self, "unit_metadata", unit_metadata)

    @classmethod
    def from_matrix(
        cls,
        *,
        candidate_ids: Sequence[str],
        unit_ids: Sequence[str],
        design: AuditDesign,
        losses: Any,
        unit_weights: Optional[Sequence[float]] = None,
        failure_mask: Optional[Any] = None,
        nonfinite: str = "raise",
        failure_loss: Optional[float] = None,
        archive: Optional[CandidateArchive] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        unit_metadata: Optional[Sequence[Mapping[str, Any]]] = None,
    ) -> "LossAudit":
        """Build an audit without candidate-specific masking.

        ``nonfinite='raise'`` is the default.  ``nonfinite='penalize'`` replaces
        all declared failures and non-finite values by one predeclared finite
        ``failure_loss``.  This ensures that a candidate cannot improve its
        estimated risk by becoming undefined on difficult units.
        """

        candidate_ids_t = tuple(str(item) for item in candidate_ids)
        unit_ids_t = tuple(str(item) for item in unit_ids)
        raw = np.asarray(losses, dtype=np.float64)
        expected_shape = (len(unit_ids_t), len(candidate_ids_t))
        if raw.shape != expected_shape:
            raise ValueError(f"losses must have shape {expected_shape!r}, got {raw.shape!r}")

        if failure_mask is None:
            failures = np.zeros(expected_shape, dtype=np.bool_)
        else:
            failures = np.asarray(failure_mask, dtype=np.bool_)
            if failures.shape != expected_shape:
                raise ValueError(
                    f"failure_mask must have shape {expected_shape!r}, got {failures.shape!r}"
                )
        failures = np.asarray(failures | ~np.isfinite(raw), dtype=np.bool_)

        policy = str(nonfinite).strip().lower()
        if policy not in {"raise", "penalize"}:
            raise ValueError("nonfinite must be either 'raise' or 'penalize'")
        cleaned = np.array(raw, dtype=np.float64, copy=True)
        resolved_failure_loss: Optional[float] = None
        if np.any(failures):
            if policy == "raise":
                positions = np.argwhere(failures)
                preview = [
                    (unit_ids_t[int(i)], candidate_ids_t[int(j)])
                    for i, j in positions[:8]
                ]
                raise ValueError(
                    "candidate failures/non-finite losses are present; either repair the "
                    f"evaluation or declare a common failure_loss. First failures: {preview!r}"
                )
            if failure_loss is None:
                raise ValueError("failure_loss is required when nonfinite='penalize'")
            resolved_failure_loss = float(failure_loss)
            if not math.isfinite(resolved_failure_loss):
                raise ValueError("failure_loss must be finite")
            valid_observed = raw[~failures & np.isfinite(raw)]
            if valid_observed.size and resolved_failure_loss < float(np.max(valid_observed)):
                raise ValueError(
                    "failure_loss must be at least as large as every observed finite loss"
                )
            cleaned[failures] = resolved_failure_loss
        elif failure_loss is not None:
            resolved_failure_loss = float(failure_loss)
            if not math.isfinite(resolved_failure_loss):
                raise ValueError("failure_loss must be finite")

        archive_fingerprint: Optional[str] = None
        if archive is not None:
            if not archive.frozen:
                raise RuntimeError("freeze the candidate archive before constructing its audit")
            archive.assert_candidate_ids(candidate_ids_t)
            archive_fingerprint = archive.fingerprint

        return cls(
            candidate_ids=candidate_ids_t,
            unit_ids=unit_ids_t,
            design=design,
            losses=cleaned,
            unit_weights=_normalised_weights(len(unit_ids_t), unit_weights),
            failure_mask=failures,
            failure_loss=resolved_failure_loss,
            archive_fingerprint=archive_fingerprint,
            metadata={} if metadata is None else metadata,
            unit_metadata=() if unit_metadata is None else tuple(unit_metadata),
        )

    @classmethod
    def from_records(
        cls,
        records: Sequence[UnitLossRecord],
        *,
        design: AuditDesign,
        candidate_ids: Optional[Sequence[str]] = None,
        unit_weights: Optional[Sequence[float]] = None,
        nonfinite: str = "raise",
        failure_loss: Optional[float] = None,
        archive: Optional[CandidateArchive] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> "LossAudit":
        if not records:
            raise ValueError("at least one UnitLossRecord is required")
        records_t = tuple(records)
        if candidate_ids is None:
            if archive is not None:
                candidate_ids_t = archive.candidate_ids
            else:
                candidate_ids_t = tuple(sorted(records_t[0].losses))
        else:
            candidate_ids_t = tuple(str(item) for item in candidate_ids)

        matrix = np.empty((len(records_t), len(candidate_ids_t)), dtype=np.float64)
        failures = np.zeros_like(matrix, dtype=np.bool_)
        for row, record in enumerate(records_t):
            if not isinstance(record, UnitLossRecord):
                raise TypeError(f"record {row} is not a UnitLossRecord")
            missing = [
                candidate_id
                for candidate_id in candidate_ids_t
                if candidate_id not in record.losses
            ]
            extra = sorted(set(record.losses) - set(candidate_ids_t))
            unknown_failures = sorted(set(record.failures) - set(candidate_ids_t))
            if missing or extra or unknown_failures:
                raise ValueError(
                    f"unit {record.unit_id!r} does not match the frozen candidate set; "
                    f"missing={missing!r}, extra={extra!r}, "
                    f"unknown_failures={unknown_failures!r}"
                )
            for col, candidate_id in enumerate(candidate_ids_t):
                matrix[row, col] = float(record.losses[candidate_id])
                failures[row, col] = candidate_id in record.failures

        return cls.from_matrix(
            candidate_ids=candidate_ids_t,
            unit_ids=[record.unit_id for record in records_t],
            design=design,
            losses=matrix,
            unit_weights=unit_weights,
            failure_mask=failures,
            nonfinite=nonfinite,
            failure_loss=failure_loss,
            archive=archive,
            metadata=metadata,
            unit_metadata=[record.metadata for record in records_t],
        )

    @property
    def n_units(self) -> int:
        return int(len(self.unit_ids))

    @property
    def n_candidates(self) -> int:
        return int(len(self.candidate_ids))

    @property
    def effective_unit_count(self) -> float:
        return float(1.0 / np.sum(self.unit_weights * self.unit_weights))

    @property
    def risks(self) -> np.ndarray:
        out = np.asarray(self.unit_weights @ self.losses, dtype=np.float64)
        out.setflags(write=False)
        return out

    @property
    def marginal_standard_errors(self) -> np.ndarray:
        centered = self.losses - self.risks[None, :]
        numerator = np.sum((self.unit_weights[:, None] * centered) ** 2, axis=0)
        correction = 1.0 - float(np.sum(self.unit_weights * self.unit_weights))
        out = np.sqrt(np.maximum(0.0, numerator / correction))
        out.setflags(write=False)
        return out

    def candidate_index(self, candidate_id: str) -> int:
        key = str(candidate_id)
        try:
            return self.candidate_ids.index(key)
        except ValueError as exc:
            raise KeyError(key) from exc

    def paired_losses(self, challenger_id: str, incumbent_id: str) -> np.ndarray:
        j = self.candidate_index(challenger_id)
        i = self.candidate_index(incumbent_id)
        out = np.asarray(self.losses[:, j] - self.losses[:, i], dtype=np.float64)
        out.setflags(write=False)
        return out

    def paired_difference(self, challenger_id: str, incumbent_id: str) -> tuple[float, float]:
        values = self.paired_losses(challenger_id, incumbent_id)
        estimate = float(self.unit_weights @ values)
        centered = values - estimate
        numerator = float(np.sum((self.unit_weights * centered) ** 2))
        correction = 1.0 - float(np.sum(self.unit_weights * self.unit_weights))
        standard_error = math.sqrt(max(0.0, numerator / correction))
        return estimate, standard_error

    def failure_counts(self) -> dict[str, int]:
        counts = np.sum(self.failure_mask, axis=0)
        return {
            candidate_id: int(counts[i])
            for i, candidate_id in enumerate(self.candidate_ids)
        }

    def assert_archive(self, archive: CandidateArchive) -> None:
        if not archive.frozen:
            raise RuntimeError("candidate archive is not frozen")
        archive.assert_candidate_ids(self.candidate_ids)
        if self.archive_fingerprint is None:
            raise ValueError("loss audit is not bound to a frozen candidate archive")
        if self.archive_fingerprint != archive.fingerprint:
            raise ValueError("loss audit was built against a different candidate archive")

    @property
    def fingerprint(self) -> str:
        digest = hashlib.sha256()
        digest.update(json.dumps(self.candidate_ids, ensure_ascii=False).encode("utf-8"))
        digest.update(json.dumps(self.unit_ids, ensure_ascii=False).encode("utf-8"))
        digest.update(canonical_json(self.design.to_dict()).encode("utf-8"))
        digest.update(np.asarray(self.unit_weights, dtype="<f8").tobytes(order="C"))
        digest.update(np.asarray(self.losses, dtype="<f8").tobytes(order="C"))
        digest.update(np.asarray(self.failure_mask, dtype=np.uint8).tobytes(order="C"))
        digest.update(str(self.archive_fingerprint).encode("utf-8"))
        return digest.hexdigest()

    def to_dict(self, *, include_losses: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "candidate_ids": list(self.candidate_ids),
            "unit_ids": list(self.unit_ids),
            "design": self.design.to_dict(),
            "unit_weights": self.unit_weights.tolist(),
            "effective_unit_count": self.effective_unit_count,
            "failure_loss": self.failure_loss,
            "failure_counts": self.failure_counts(),
            "archive_fingerprint": self.archive_fingerprint,
            "audit_fingerprint": self.fingerprint,
            "metadata": thaw_json(self.metadata),
            "unit_metadata": [thaw_json(item) for item in self.unit_metadata],
            "risks": self.risks.tolist(),
            "marginal_standard_errors": self.marginal_standard_errors.tolist(),
        }
        if include_losses:
            payload["losses"] = self.losses.tolist()
            payload["failure_mask"] = self.failure_mask.tolist()
        return payload
