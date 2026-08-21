# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""CoE slice-committee diagnostics.

Wave 1 is deliberately final-only: it evaluates already-visible analytic
candidate expressions on independent deterministic slices and reports a
noise-aware committee summary.  It does not mutate Stage A/B state.
"""

from __future__ import annotations

import copy
import json
import math
import os
import re
from dataclasses import asdict, dataclass, field, replace
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None

from nestynet_sr.equation_polisher import (
    PolishConfig,
    _eval_expr_array,
    _mse_and_se,
    expression_complexity,
    has_unsupported_artifact_call,
    infer_variable_names,
    parse_sympy_expr,
)
from nestynet_sr.sr_core.coefficient_metadata import (
    coefficient_symbol_values,
    coefficient_symbol_values_for_expression,
    normalize_coefficient_metadata_by_dataset,
)

_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


@dataclass(frozen=True)
class SliceSpec:
    slice_id: int
    train_start: int
    train_stop: int
    val_start: int
    val_stop: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CandidateArtifact:
    candidate_id: str
    expr: str
    source: str
    label: str = ""
    complexity: Optional[float] = None
    n_free_params: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ProposalReservoir:
    """Dedupe portable analytic proposal expressions for final CoE scoring."""

    max_candidates: int = 32
    _entries: dict[str, dict[str, Any]] = field(default_factory=dict)

    def add_expr(
        self,
        expr: Any,
        *,
        source: str,
        label: str = "",
        complexity: Optional[float] = None,
        n_free_params: int = 0,
        loss: Optional[float] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> bool:
        text = _clean_expr(expr)
        if text is None:
            return False
        meta = dict(metadata or {})
        key = _candidate_key(text, meta.get("coefficient_metadata"))
        entry = self._entries.get(key)
        loss_f: Optional[float]
        try:
            loss_f = float(loss) if loss is not None else None
            if loss_f is not None and not math.isfinite(loss_f):
                loss_f = None
        except Exception:
            loss_f = None
        cx_f: Optional[float]
        try:
            cx_f = float(complexity) if complexity is not None else None
            if cx_f is not None and not math.isfinite(cx_f):
                cx_f = None
        except Exception:
            cx_f = None
        if entry is None:
            self._entries[key] = {
                "expr": text,
                "source": str(source),
                "label": str(label or source),
                "complexity": cx_f,
                "n_free_params": max(0, int(n_free_params or 0)),
                "support_count": 1,
                "sources": [str(source)],
                "best_loss": loss_f,
                "metadata": meta,
            }
            return True
        sources = list(entry.get("sources") or [])
        if str(source) not in sources:
            sources.append(str(source))
            entry["support_count"] = int(entry.get("support_count", 1) or 1) + 1
        entry["sources"] = sources
        if cx_f is not None:
            old_cx = entry.get("complexity")
            if old_cx is None or cx_f < float(old_cx):
                entry["complexity"] = cx_f
        if loss_f is not None:
            old_loss = entry.get("best_loss")
            if old_loss is None or loss_f < float(old_loss):
                entry["best_loss"] = loss_f
                entry["source"] = str(source)
                entry["label"] = str(label or source)
        merged_meta = dict(entry.get("metadata") or {})
        merged_meta.update(meta)
        entry["metadata"] = merged_meta
        return True

    def entries(self) -> list[dict[str, Any]]:
        def _sort_key(row: dict[str, Any]) -> tuple[int, float, float, str]:
            loss = row.get("best_loss")
            cx = row.get("complexity")
            return (
                -int(row.get("support_count", 1) or 1),
                float(loss) if loss is not None else float("inf"),
                float(cx) if cx is not None else float("inf"),
                str(row.get("expr", "")),
            )

        return sorted(self._entries.values(), key=_sort_key)[
            : max(1, int(self.max_candidates or 1))
        ]

    def to_dict(self) -> dict[str, Any]:
        rows = []
        for idx, row in enumerate(self.entries()):
            payload = dict(row)
            payload["reservoir_id"] = f"r{idx:03d}"
            rows.append(payload)
        return {
            "enabled": True,
            "total_unique": int(len(self._entries)),
            "max_candidates": int(self.max_candidates),
            "candidates": rows,
        }


@dataclass
class StageAProposalReservoir:
    """Dedupe portable Stage-A proposal records from reference/scout slices.

    These are not accepted histories. They are visible proposal artifacts that a
    reference run may later materialize and adjudicate.
    """

    max_candidates: int = 64
    _entries: dict[str, dict[str, Any]] = field(default_factory=dict)

    def add_proposal(
        self,
        *,
        kind: str,
        payload: Any,
        source: str,
        label: str = "",
        score: Optional[float] = None,
        loss: Optional[float] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> bool:
        kind_s = str(kind or "").strip()
        payload_c = _canonical_stageA_payload(payload)
        if not kind_s or payload_c is None:
            return False
        key = _stageA_proposal_key(kind_s, payload_c)
        score_f = _as_finite_float(score)
        loss_f = _as_finite_float(loss)
        meta = dict(metadata or {})
        entry = self._entries.get(key)
        if entry is None:
            self._entries[key] = {
                "kind": kind_s,
                "payload": payload_c,
                "source": str(source),
                "label": str(label or kind_s),
                "score": score_f,
                "best_loss": loss_f,
                "support_count": 1,
                "sources": [str(source)],
                "metadata": meta,
            }
            return True

        sources = list(entry.get("sources") or [])
        if str(source) not in sources:
            sources.append(str(source))
            entry["support_count"] = int(entry.get("support_count", 1) or 1) + 1
        entry["sources"] = sources
        if score_f is not None:
            old_score = entry.get("score")
            if old_score is None or score_f > float(old_score):
                entry["score"] = score_f
                entry["source"] = str(source)
                entry["label"] = str(label or kind_s)
        if loss_f is not None:
            old_loss = entry.get("best_loss")
            if old_loss is None or loss_f < float(old_loss):
                entry["best_loss"] = loss_f
        merged_meta = dict(entry.get("metadata") or {})
        merged_meta.update(meta)
        entry["metadata"] = merged_meta
        return True

    def entries(self) -> list[dict[str, Any]]:
        def _sort_key(row: dict[str, Any]) -> tuple[int, float, float, str]:
            loss = row.get("best_loss")
            score = row.get("score")
            return (
                -int(row.get("support_count", 1) or 1),
                -(float(score) if score is not None else float("-inf")),
                float(loss) if loss is not None else float("inf"),
                str(row.get("kind", "")) + ":" + json.dumps(row.get("payload"), sort_keys=True),
            )

        return sorted(self._entries.values(), key=_sort_key)[
            : max(1, int(self.max_candidates or 1))
        ]

    def to_dict(self) -> dict[str, Any]:
        rows = []
        for idx, row in enumerate(self.entries()):
            payload = copy.deepcopy(row)
            payload["reservoir_id"] = f"a{idx:03d}"
            rows.append(payload)
        return {
            "enabled": True,
            "kind": "stageA_proposal_reservoir",
            "total_unique": int(len(self._entries)),
            "max_candidates": int(self.max_candidates),
            "candidates": rows,
        }


@dataclass(frozen=True)
class SliceFitResult:
    candidate_id: str
    slice_id: int
    val_mse: float
    val_mse_se: float
    frac_valid: float
    n_val: int
    status: str = "success"
    error: Optional[str] = None
    #: Per-row squared errors aligned with the slice's ``[val_start, val_stop)``
    #: rows (NaN where the prediction or target is non-finite).  Only populated
    #: when the evaluation is asked for it; the paired-row committee inference
    #: consumes these so slices can stay pure compute partitions while the rows
    #: are the statistical units.
    row_losses: Optional[tuple[float, ...]] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CommitteeDecision:
    mode: str
    status: str
    incumbent_id: Optional[str]
    recommended_id: Optional[str]
    recommended_expr: Optional[str]
    n_candidates: int
    n_slices: int
    candidate_summary: list[dict[str, Any]]
    slice_specs: list[dict[str, Any]]
    results: list[dict[str, Any]]
    config: dict[str, Any]
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CommitteeEvalCache:
    """Small in-process cache for fixed-expression slice evaluations."""

    enabled: bool = True
    _rows: dict[tuple[Any, ...], SliceFitResult] = field(default_factory=dict)
    hits: int = 0
    misses: int = 0

    def _key(
        self,
        cand: CandidateArtifact,
        *,
        filepath: str | Path,
        spec: SliceSpec,
        min_valid_fraction: float,
    ) -> tuple[Any, ...]:
        return (
            str(Path(filepath).expanduser().resolve()),
            _candidate_key(
                str(cand.expr),
                _candidate_coefficient_metadata(cand),
            ),
            int(spec.val_start),
            int(spec.val_stop),
            round(float(min_valid_fraction), 8),
        )

    def evaluate(
        self,
        cand: CandidateArtifact,
        *,
        filepath: str | Path,
        spec: SliceSpec,
        min_valid_fraction: float = 0.80,
        return_row_losses: bool = False,
    ) -> SliceFitResult:
        if not bool(self.enabled):
            self.misses += 1
            return evaluate_candidate_on_slice(
                cand,
                filepath=filepath,
                spec=spec,
                min_valid_fraction=min_valid_fraction,
                return_row_losses=return_row_losses,
            )
        # The flag is part of the key: a cached aggregate-only result must not
        # satisfy a request that needs the per-row losses.
        key = self._key(
            cand,
            filepath=filepath,
            spec=spec,
            min_valid_fraction=min_valid_fraction,
        ) + (bool(return_row_losses),)
        if key in self._rows:
            self.hits += 1
            return replace(self._rows[key], candidate_id=cand.candidate_id)
        self.misses += 1
        res = evaluate_candidate_on_slice(
            cand,
            filepath=filepath,
            spec=spec,
            min_valid_fraction=min_valid_fraction,
            return_row_losses=return_row_losses,
        )
        self._rows[key] = replace(res, candidate_id="__cached__")
        return res

    def stats(self) -> dict[str, Any]:
        total = int(self.hits + self.misses)
        return {
            "enabled": bool(self.enabled),
            "entries": int(len(self._rows)),
            "hits": int(self.hits),
            "misses": int(self.misses),
            "requests": total,
            "hit_fraction": float(self.hits / total) if total > 0 else 0.0,
        }

    def snapshot(self) -> dict[str, SliceFitResult]:
        return copy.copy(self._rows)


def evaluate_candidate_on_slice_cached(
    cand: CandidateArtifact,
    *,
    filepath: str | Path,
    spec: SliceSpec,
    min_valid_fraction: float = 0.80,
    cache: Optional[CommitteeEvalCache] = None,
    return_row_losses: bool = False,
) -> SliceFitResult:
    if cache is None:
        return evaluate_candidate_on_slice(
            cand,
            filepath=filepath,
            spec=spec,
            min_valid_fraction=min_valid_fraction,
            return_row_losses=return_row_losses,
        )
    return cache.evaluate(
        cand,
        filepath=filepath,
        spec=spec,
        min_valid_fraction=min_valid_fraction,
        return_row_losses=return_row_losses,
    )


def _clean_expr_verbose(value: Any) -> tuple[Optional[str], Optional[str]]:
    """Return ``(portable_text, drop_reason)``; exactly one side is ``None``."""
    if value is None:
        return None, "empty"
    text = _ANSI_ESCAPE_RE.sub("", str(value))
    text = _CONTROL_CHAR_RE.sub("", text).strip()
    if not text or text.lower() in {"none", "nan", "null"}:
        return None, "empty"
    text_lower = text.lower()
    if "NN[" in text or "leaf" in text_lower or "nn" in text_lower:
        return None, "unresolved_nn_leaf"
    # Use the same call whitelist as the equation polisher.  Unknown calls are
    # fitted/local artifacts, not portable SymPy functions.
    if has_unsupported_artifact_call(text):
        return None, "local_fitted_wrapper"
    return text, None


def _clean_expr(value: Any) -> Optional[str]:
    text, _reason = _clean_expr_verbose(value)
    return text


def _coefficient_value_fingerprint(payload: Any) -> str:
    if payload is None:
        return "no-coefficient-metadata"
    try:
        values = coefficient_symbol_values(payload)
        normalized = [
            [name, float(value).hex()] for name, value in sorted(values.items())
        ]
        return json.dumps(normalized, separators=(",", ":"))
    except Exception:
        try:
            return "invalid:" + json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            )
        except Exception:
            return "invalid:" + repr(payload)


def _candidate_key(expr: str, coefficient_metadata: Any = None) -> str:
    expression_key = " ".join(str(expr).strip().split())
    return expression_key + "\x1f" + _coefficient_value_fingerprint(
        coefficient_metadata
    )


def _candidate_coefficient_metadata(cand: CandidateArtifact) -> Any:
    metadata = cand.metadata if isinstance(cand.metadata, dict) else {}
    return metadata.get("coefficient_metadata")


def _selection_n_free_params(rec: Mapping[str, Any]) -> int:
    """Free-parameter count of a polish row for statistical selection.

    Prefers the row's explicit frozen-selection declaration
    (``selection_n_free_params``) over the polish frontier's conservative
    display count, so proposal batteries that fit literals on search data and
    freeze them before the archive is built are charged like Stage-B/C rows
    (whose fitted-and-frozen constants declare 0 free parameters).
    """

    value = rec.get("selection_n_free_params")
    if value is None:
        value = rec.get("n_free_params") or 0
    try:
        return max(0, int(value))
    except Exception:
        return 0


def _metadata_with_coefficient_bundle(
    metadata: Optional[dict[str, Any]],
    record: Optional[dict[str, Any]] = None,
    *,
    fallback: Any = None,
) -> dict[str, Any]:
    out = dict(metadata or {})
    coefficient_metadata = None
    coefficient_metadata_by_dataset = None
    dataset_ids = None
    if isinstance(record, dict):
        coefficient_metadata = record.get("coefficient_metadata")
        coefficient_metadata_by_dataset = record.get(
            "coefficient_metadata_by_dataset"
        )
        dataset_ids = record.get("dataset_ids")
        nested = record.get("metadata")
        if coefficient_metadata is None and isinstance(nested, dict):
            coefficient_metadata = nested.get("coefficient_metadata")
        if coefficient_metadata_by_dataset is None and isinstance(nested, dict):
            coefficient_metadata_by_dataset = nested.get(
                "coefficient_metadata_by_dataset"
            )
        if dataset_ids is None and isinstance(nested, dict):
            dataset_ids = nested.get("dataset_ids")
    if coefficient_metadata is None:
        coefficient_metadata = fallback
    if coefficient_metadata_by_dataset is not None:
        normalized_by_dataset = normalize_coefficient_metadata_by_dataset(
            coefficient_metadata_by_dataset,
            primary_payload=coefficient_metadata,
            expected_dataset_ids=dataset_ids,
        )
        if normalized_by_dataset:
            coefficient_metadata = normalized_by_dataset[0]
    if coefficient_metadata is not None:
        out["coefficient_metadata"] = copy.deepcopy(coefficient_metadata)
    if coefficient_metadata_by_dataset is not None:
        out["coefficient_metadata_by_dataset"] = copy.deepcopy(
            coefficient_metadata_by_dataset
        )
    if dataset_ids is not None:
        out["dataset_ids"] = copy.deepcopy(dataset_ids)
    return out


_FLOAT_LITERAL_RE = re.compile(
    r"(?<![A-Za-z_0-9])[-+]?"
    r"(?:(?:\d+\.\d*)|(?:\.\d+)|(?:\d+[eE][-+]?\d+))"
    r"(?:[eE][-+]?\d+)?"
)


def _nontrivial_float_literal_count(expr: Any) -> int:
    """Count fitted-looking decimal constants in a rendered expression."""

    text = str(expr or "")
    count = 0
    for m in _FLOAT_LITERAL_RE.finditer(text):
        raw = m.group(0)
        try:
            val = float(raw)
        except Exception:
            continue
        if not math.isfinite(val):
            continue
        # Formatting artefacts such as 1.0 or 0.0 should not make an otherwise
        # exact symbolic expression lose a noisy tie to a decimal-fitted one.
        if abs(val - round(val)) <= 1.0e-12 * max(1.0, abs(val)):
            continue
        count += 1
    return int(count)


def _symbolic_exactness_adjustment(cand_payload: dict[str, Any]) -> float:
    """Small noisy-tie complexity adjustment for fitted-looking constants."""

    metadata = cand_payload.get("metadata") if isinstance(cand_payload, dict) else None
    metadata = metadata if isinstance(metadata, dict) else {}
    try:
        float_count = int(metadata.get("float_literal_count", 0) or 0)
    except Exception:
        float_count = 0
    penalty = 8.0 * max(0, float_count)
    expr = str(cand_payload.get("expr", "") if isinstance(cand_payload, dict) else "")
    if float_count == 0 and any(tok in expr for tok in ("pi", "E")):
        penalty -= 1.0
    return float(max(-1.0, penalty))


def _stagec_expression_certified(sympy_meta: Any) -> bool:
    """Return whether a Stage-C string is safe enough for final CoE scoring."""

    if not isinstance(sympy_meta, dict) or not sympy_meta:
        return True
    unit_certificate = sympy_meta.get("unit_admissibility")
    if isinstance(unit_certificate, dict):
        if str(unit_certificate.get("code") or "") == "expression_unavailable":
            return False
        if (
            unit_certificate.get("checked") is True
            and unit_certificate.get("valid") is False
        ):
            return False
        if (
            sympy_meta.get("accepted") is True
            and unit_certificate.get("checked") is True
            and unit_certificate.get("valid") is not True
        ):
            return False
    if (
        sympy_meta.get("units_checked") is True
        and sympy_meta.get("units_ok") is False
    ):
        return False
    if (
        sympy_meta.get("accepted") is True
        and sympy_meta.get("units_checked") is True
        and sympy_meta.get("units_ok") is not True
    ):
        return False
    kind = str(sympy_meta.get("kind", "") or "").lower()
    if kind in {
        "bad_pretty_print",
        "unsafe_pretty_print",
        "uncertified",
        "unit_check_expression_unavailable",
    }:
        return False
    if sympy_meta.get("accepted") is True or sympy_meta.get("verified") is True:
        return True
    # Older reports sometimes omitted these fields.  If they are present and
    # false, respect that; if neither is present, keep legacy behavior.
    if "accepted" in sympy_meta or "verified" in sympy_meta:
        return False
    return True


def _canonical_stageA_payload(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for k, v in sorted(value.items(), key=lambda kv: str(kv[0])):
            vv = _canonical_stageA_payload(v)
            if vv is not None:
                out[str(k)] = vv
        return out if out else None
    if isinstance(value, (list, tuple, set)):
        seq = list(value)
        if isinstance(value, set):
            seq = sorted(seq, key=str)
        return [_canonical_stageA_payload(v) for v in seq]
    if isinstance(value, (str, bool)) or value is None:
        text = str(value).strip() if isinstance(value, str) else value
        return text if text != "" else None
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        f = float(value)
        return f if math.isfinite(f) else None
    try:
        text = str(value).strip()
    except Exception:
        return None
    return text if text else None


def _stageA_proposal_key(kind: str, payload: Any) -> str:
    identity_payload = payload
    if str(kind) == "compound_coordinate_replay" and isinstance(payload, dict):
        replay_key = payload.get("replay_key")
        if isinstance(replay_key, dict):
            identity_payload = _canonical_stageA_payload(replay_key)
    try:
        payload_s = json.dumps(identity_payload, sort_keys=True, separators=(",", ":"))
    except Exception:
        payload_s = str(identity_payload)
    return str(kind) + ":" + payload_s


def _add_candidate(
    out: list[CandidateArtifact],
    seen: Optional[set[str]],
    *,
    expr: Any,
    source: str,
    label: str = "",
    complexity: Optional[float] = None,
    n_free_params: int = 0,
    metadata: Optional[dict[str, Any]] = None,
    dropped_log: Optional[list[dict[str, Any]]] = None,
) -> None:
    text, drop_reason = _clean_expr_verbose(expr)
    if text is None:
        _append_expression_drop(
            dropped_log,
            expr=expr,
            reason=drop_reason,
            source=source,
            label=label,
        )
        return
    meta = dict(metadata or {})
    key = _candidate_key(text, meta.get("coefficient_metadata"))
    if seen is not None:
        if key in seen:
            return
        seen.add(key)
    meta.setdefault("float_literal_count", _nontrivial_float_literal_count(text))
    meta.setdefault("has_float_literals", bool(int(meta.get("float_literal_count", 0) or 0) > 0))
    cid = f"c{len(out):03d}"
    out.append(
        CandidateArtifact(
            candidate_id=cid,
            expr=text,
            source=str(source),
            label=str(label or source),
            complexity=None if complexity is None else float(complexity),
            n_free_params=max(0, int(n_free_params or 0)),
            metadata=meta,
        )
    )


def _append_expression_drop(
    dropped_log: Optional[list[dict[str, Any]]],
    *,
    expr: Any,
    reason: Optional[str],
    source: str,
    label: str,
) -> None:
    if dropped_log is None or reason in {None, "empty"}:
        return
    preview = _CONTROL_CHAR_RE.sub("", _ANSI_ESCAPE_RE.sub("", str(expr))).strip()
    dropped_log.append(
        {
            "reason": str(reason),
            "source": str(source),
            "label": str(label or source),
            "expr_preview": preview[:120],
        }
    )


def _uncertified_stagec_drop_reason(expr: Any, sympy_meta: dict[str, Any]) -> Optional[str]:
    text, reason = _clean_expr_verbose(expr)
    if text is None:
        return reason
    unit_certificate = sympy_meta.get("unit_admissibility")
    if isinstance(unit_certificate, dict):
        if unit_certificate.get("checked") is True and unit_certificate.get("valid") is False:
            return "unit_invalid"
    if sympy_meta.get("units_checked") is True and sympy_meta.get("units_ok") is False:
        return "unit_invalid"
    return "stagec_uncertified"


def _as_finite_float(value: Any) -> Optional[float]:
    try:
        f = float(value)
    except Exception:
        return None
    if not math.isfinite(f):
        return None
    return float(f)


def _is_identity_y_transform_name(value: Any) -> bool:
    text = str(value or "identity").strip().lower()
    return text in {"", "identity", "none", "null"}


def _stageB_path_expr_for_raw_y_reservoir(
    step: dict[str, Any],
    *,
    selected_y_transform: Any,
) -> tuple[Optional[Any], str]:
    """Return a raw-y expression from a path step, or ``None`` for φ-space rows."""
    if not isinstance(step, dict):
        return None, "invalid"
    y_name = step.get("y_transform", selected_y_transform)
    if not _is_identity_y_transform_name(y_name):
        expr = step.get("y_expression") or step.get("y_expr") or step.get("raw_y_expr")
        if expr is None:
            return None, "phi"
        return expr, "raw_y"
    return (
        step.get("y_expression")
        or step.get("expression")
        or step.get("expr")
        or step.get("y_expr"),
        "raw_y",
    )


def build_stageB_proposal_reservoir(
    *,
    decision_log: Optional[Sequence[dict[str, Any]]] = None,
    simplification_path: Optional[Sequence[dict[str, Any]]] = None,
    stageB_data: Optional[dict[str, Any]] = None,
    max_candidates: int = 32,
) -> dict[str, Any]:
    """Build a portable reservoir from visible analytic Stage-B expressions.

    The reservoir intentionally stores expression strings, not AST/model state.
    Non-portable candidates with unresolved NN/leaf placeholders are ignored by
    ``_clean_expr`` and will not enter committee scoring.
    """
    reservoir = ProposalReservoir(max_candidates=max(1, int(max_candidates or 1)))
    selected_y_transform = None
    if isinstance(stageB_data, dict):
        selected_y_transform = stageB_data.get("y_selected")
    selected_is_identity = _is_identity_y_transform_name(selected_y_transform)

    for rec in list(decision_log or []):
        if not isinstance(rec, dict):
            continue
        if not selected_is_identity:
            # Decision-log snapshots are AST/φ-space expressions for transformed
            # branches.  Raw-y final candidates are exported through branch
            # artifacts and stageB_final:y_expr below.
            continue
        expr = rec.get("ast_snapshot") or rec.get("candidate_expr") or rec.get("expr")
        if expr is None:
            continue
        outcome = str(rec.get("outcome", "unknown") or "unknown")
        rule = str(rec.get("rule", "") or "")
        label = str(rec.get("label", "") or rule or outcome)
        reservoir.add_expr(
            expr,
            source=f"stageB_decision:{outcome}:{rule or 'unknown'}",
            label=label,
            complexity=_as_finite_float(
                rec.get("cand_complexity_total")
                if rec.get("cand_complexity_total") is not None
                else rec.get("cand_ast_cost")
            ),
            n_free_params=0,
            loss=_as_finite_float(rec.get("cand_loss")),
            metadata=_metadata_with_coefficient_bundle(
                {
                    "outcome": outcome,
                    "rule": rule,
                    "reason": rec.get("reason"),
                    "pattern_family": rec.get("pattern_family"),
                    "decision_step": rec.get("step"),
                    "expression_space": "raw_y",
                },
                rec,
            ),
        )

    for step in list(simplification_path or []):
        if not isinstance(step, dict):
            continue
        expr, expr_space = _stageB_path_expr_for_raw_y_reservoir(
            step,
            selected_y_transform=selected_y_transform,
        )
        if expr is None:
            continue
        reservoir.add_expr(
            expr,
            source=f"simplification_path:{step.get('stage', '')}",
            label=step.get("action") or "simplification_path",
            complexity=_as_finite_float(step.get("ast_cost")),
            n_free_params=0,
            loss=_as_finite_float(
                step.get("mse_raw")
                if step.get("mse_raw") is not None and expr_space == "raw_y"
                else step.get("val_loss")
            ),
            metadata=_metadata_with_coefficient_bundle(
                {
                    "step": step.get("step"),
                    "stage": step.get("stage"),
                    "expression_space": expr_space,
                    "y_transform": step.get("y_transform", selected_y_transform),
                },
                step,
            ),
        )

    if isinstance(stageB_data, dict):
        for rec in list(stageB_data.get("y_branch_artifacts") or []):
            if not isinstance(rec, dict):
                continue
            reservoir.add_expr(
                rec.get("expr"),
                source=f"stageB_y_branch:{rec.get('label', 'branch')}",
                label=rec.get("label") or "stageB_y_branch",
                complexity=_as_finite_float(rec.get("complexity")),
                n_free_params=max(0, int(rec.get("n_free_params") or 0)),
                loss=_as_finite_float(rec.get("loss")),
                metadata=_metadata_with_coefficient_bundle(
                    {
                        **dict(rec.get("metadata") or {}),
                        "expression_space": "raw_y",
                    },
                    rec,
                ),
            )
        reservoir.add_expr(
            stageB_data.get("y_expr_raw_str") or stageB_data.get("y_expr_str"),
            source="stageB_final:y_expr",
            label="stageB_final",
            loss=_as_finite_float(
                stageB_data.get("original_y_val_loss")
                if stageB_data.get("original_y_val_loss") is not None
                else stageB_data.get("val_loss")
            ),
            metadata=_metadata_with_coefficient_bundle(
                {"final_stageB": True, "expression_space": "raw_y"},
                stageB_data,
            ),
        )

    payload = reservoir.to_dict()
    payload["source"] = "stageB_reference_run"
    return payload


def build_stageA_proposal_reservoir(
    *,
    stageA_data: Optional[dict[str, Any]] = None,
    max_candidates: int = 64,
    source: str = "stageA_reference_run",
) -> dict[str, Any]:
    """Build a portable Stage-A proposal reservoir from accepted move records.

    The payload stores visible transactions and proposal descriptors only.  It
    does not store fitted NN parameters or accepted scout history.
    """
    reservoir = StageAProposalReservoir(max_candidates=max(1, int(max_candidates or 1)))
    if not isinstance(stageA_data, dict):
        payload = reservoir.to_dict()
        payload["source"] = str(source)
        return payload

    y_name = str(stageA_data.get("y_op_name", "identity") or "identity")
    base_meta = {
        "stageA_status": stageA_data.get("stageA_status"),
        "y_transform": y_name,
        "fit_y_link": stageA_data.get("fit_y_link"),
    }

    ybranch = stageA_data.get("coe_stageA_ybranch_committee")
    if isinstance(ybranch, dict):
        for row in list(ybranch.get("branches") or []):
            if not isinstance(row, dict):
                continue
            branch_id = str(row.get("branch_id") or "").strip()
            if not branch_id or branch_id == "identity":
                continue
            reservoir.add_proposal(
                kind="y_branch",
                payload={"y_transform": branch_id},
                source=f"{source}:y_branch:{branch_id}",
                label=str(row.get("branch_id") or "y_branch"),
                score=float(row.get("wins", 0) or 0) + 0.5 * float(row.get("ties", 0) or 0),
                loss=_as_finite_float(row.get("median_raw_y_mse")),
                metadata={**base_meta, "allowed": row.get("allowed"), "losses": row.get("losses")},
            )

    def _add_split(kind: str, partition: Any, *, label: str, score: Any = None, metadata: Optional[dict[str, Any]] = None) -> None:
        payload = {"kind": str(kind), "partition": _canonical_stageA_payload(partition)}
        reservoir.add_proposal(
            kind="split_partition",
            payload=payload,
            source=f"{source}:split:{kind}",
            label=label,
            score=_as_finite_float(score),
            loss=_as_finite_float(stageA_data.get("val_loss")),
            metadata={**base_meta, **dict(metadata or {})},
        )

    if stageA_data.get("rest_add") is not None:
        _add_split("add", stageA_data.get("rest_add"), label="Stage-A additive split")
    if stageA_data.get("rest_mult") is not None:
        _add_split("mul", stageA_data.get("rest_mult"), label="Stage-A multiplicative split")

    for move in list(stageA_data.get("stageA_move_records") or []):
        if not isinstance(move, dict):
            continue
        move_kind = str(move.get("move_kind", "unknown") or "unknown")
        details = move.get("details") if isinstance(move.get("details"), dict) else {}
        risk_tags = list(move.get("risk_tags") or [])
        common_meta = {
            **base_meta,
            "move_kind": move_kind,
            "risk_tags": risk_tags,
            "move_seq": move.get("seq"),
            "parent_loss": move.get("parent_loss"),
            "candidate_loss": move.get("candidate_loss"),
        }
        if move_kind == "separability_split" or "split_accept" in risk_tags:
            op = str(details.get("op") or "")
            split_kind = "mul" if "mul" in op.lower() else "add"
            reservoir.add_proposal(
                kind="split_partition",
                payload={
                    "kind": split_kind,
                    "group1": _canonical_stageA_payload(details.get("group1")),
                    "group2": _canonical_stageA_payload(details.get("group2")),
                    "has_overlap": bool(details.get("has_overlap", False)),
                },
                source=f"{source}:move:{move.get('seq')}:split",
                label=f"Stage-A {split_kind} split",
                score=_as_finite_float(details.get("split_score")),
                loss=_as_finite_float(move.get("candidate_loss")),
                metadata={**common_meta, "details": details},
            )
        if "terminal_closure" in risk_tags:
            expr = _clean_expr(move.get("candidate_ast_human"))
            if expr is not None:
                reservoir.add_proposal(
                    kind="terminal_expression",
                    payload={"expr": expr},
                    source=f"{source}:move:{move.get('seq')}:terminal",
                    label=move_kind,
                    score=1.0,
                    loss=_as_finite_float(move.get("candidate_loss")),
                    metadata=common_meta,
                )
        if "compound_coordinate" in risk_tags:
            replay_descriptor = details.get("compound_replay_descriptor")
            if isinstance(replay_descriptor, dict) and bool(
                replay_descriptor.get("replay_eligible", False)
            ):
                reservoir.add_proposal(
                    kind="compound_coordinate_replay",
                    payload=replay_descriptor,
                    source=f"{source}:move:{move.get('seq')}:compound_replay",
                    label=f"{move_kind}:replay",
                    score=_as_finite_float(
                        replay_descriptor.get("source_evidence", {}).get("confidence")
                        if isinstance(replay_descriptor.get("source_evidence"), dict)
                        else None
                    ),
                    loss=_as_finite_float(move.get("candidate_loss")),
                    metadata={**common_meta, "details": details, "portable_replay": True},
                )
            snapshot = move.get("candidate_ast_human")
            if snapshot:
                reservoir.add_proposal(
                    kind="visible_stageA_transaction",
                    payload={
                        "move_kind": move_kind,
                        "candidate_snapshot": str(snapshot),
                        "full_compound": bool(details.get("full_compound", False)),
                    },
                    source=f"{source}:move:{move.get('seq')}:compound",
                    label=move_kind,
                    score=_as_finite_float(details.get("split_score")),
                    loss=_as_finite_float(move.get("candidate_loss")),
                    metadata=common_meta,
                )

    exit_audit = stageA_data.get("coe_stageA_exit_audit")
    if isinstance(exit_audit, dict):
        expr = _clean_expr(exit_audit.get("candidate_expr"))
        if expr is not None:
            reservoir.add_proposal(
                kind="terminal_expression",
                payload={"expr": expr},
                source=f"{source}:stageA_exit",
                label="Stage-A exit expression",
                score=1.0,
                loss=_as_finite_float(exit_audit.get("candidate_median_val_mse")),
                metadata={**base_meta, "exit_audit": True},
            )

    payload = reservoir.to_dict()
    payload["source"] = str(source)
    payload["stageA_status"] = stageA_data.get("stageA_status")
    payload["y_transform"] = y_name
    return payload


def merge_proposal_reservoir_payloads(
    payloads: Sequence[dict[str, Any]],
    *,
    max_candidates: int = 32,
) -> dict[str, Any]:
    reservoir = ProposalReservoir(max_candidates=max(1, int(max_candidates or 1)))
    sources: list[str] = []
    for payload in list(payloads or []):
        if not isinstance(payload, dict):
            continue
        src_payload = str(payload.get("source", "proposal_reservoir") or "proposal_reservoir")
        if src_payload not in sources:
            sources.append(src_payload)
        for rec in list(payload.get("candidates") or []):
            if not isinstance(rec, dict):
                continue
            rec_sources = list(rec.get("sources") or [])
            if not rec_sources:
                rec_sources = [str(rec.get("source") or src_payload)]
            for rec_source in rec_sources:
                reservoir.add_expr(
                    rec.get("expr"),
                    source=str(rec_source or rec.get("source") or src_payload),
                    label=str(rec.get("label") or "proposal_reservoir"),
                    complexity=rec.get("complexity"),
                    n_free_params=max(0, int(rec.get("n_free_params") or 0)),
                    loss=rec.get("best_loss"),
                    metadata=dict(rec.get("metadata") or {}),
                )
    out = reservoir.to_dict()
    out["source"] = "merged_proposal_reservoir"
    out["input_sources"] = sources
    return out


def merge_stageA_proposal_reservoir_payloads(
    payloads: Sequence[dict[str, Any]],
    *,
    max_candidates: int = 64,
) -> dict[str, Any]:
    reservoir = StageAProposalReservoir(max_candidates=max(1, int(max_candidates or 1)))
    sources: list[str] = []
    for payload in list(payloads or []):
        if not isinstance(payload, dict):
            continue
        src_payload = str(payload.get("source", "stageA_proposal_reservoir") or "stageA_proposal_reservoir")
        if src_payload not in sources:
            sources.append(src_payload)
        for rec in list(payload.get("candidates") or []):
            if not isinstance(rec, dict):
                continue
            rec_sources = list(rec.get("sources") or [])
            if not rec_sources:
                rec_sources = [str(rec.get("source") or src_payload)]
            for rec_source in rec_sources:
                reservoir.add_proposal(
                    kind=str(rec.get("kind") or "stageA_proposal"),
                    payload=rec.get("payload"),
                    source=str(rec_source or rec.get("source") or src_payload),
                    label=str(rec.get("label") or rec.get("kind") or "stageA_proposal"),
                    score=rec.get("score"),
                    loss=rec.get("best_loss"),
                    metadata=dict(rec.get("metadata") or {}),
                )
    out = reservoir.to_dict()
    out["source"] = "merged_stageA_proposal_reservoir"
    out["input_sources"] = sources
    return out


def stageA_terminal_proposals_as_expression_reservoir(
    payload: dict[str, Any],
    *,
    max_candidates: int = 32,
) -> dict[str, Any]:
    """Convert portable Stage-A terminal-expression proposals to final candidates."""
    reservoir = ProposalReservoir(max_candidates=max(1, int(max_candidates or 1)))
    if not isinstance(payload, dict):
        out = reservoir.to_dict()
        out["source"] = "stageA_terminal_proposal_reservoir"
        return out
    src_payload = str(payload.get("source", "stageA_proposal_reservoir") or "stageA_proposal_reservoir")
    for rec in list(payload.get("candidates") or []):
        if not isinstance(rec, dict) or str(rec.get("kind", "")) != "terminal_expression":
            continue
        rec_sources = list(rec.get("sources") or [])
        if not rec_sources:
            rec_sources = [str(rec.get("source") or src_payload)]
        rec_payload = rec.get("payload") if isinstance(rec.get("payload"), dict) else {}
        expr = rec_payload.get("expr")
        for rec_source in rec_sources:
            reservoir.add_expr(
                expr,
                source=f"stageA_terminal:{rec_source}",
                label=str(rec.get("label") or "stageA_terminal"),
                loss=rec.get("best_loss"),
                metadata={
                    **dict(rec.get("metadata") or {}),
                    "stageA_proposal_kind": rec.get("kind"),
                    "stageA_support_count": rec.get("support_count"),
                },
            )
    out = reservoir.to_dict()
    out["source"] = "stageA_terminal_proposal_reservoir"
    out["input_source"] = src_payload
    return out


def stageA_y_branch_names_from_proposal_reservoir(
    payload: dict[str, Any],
    *,
    min_support: int = 1,
) -> list[str]:
    """Return portable y-transform branch names from a Stage-A reservoir.

    The reservoir stores scout/reference observations only; this helper extracts
    the branch names that a reference run may explicitly materialize in its own
    y-search portfolio.
    """
    if not isinstance(payload, dict):
        return []
    out: list[str] = []
    seen: set[str] = set()
    support_min = max(1, int(min_support or 1))
    for rec in list(payload.get("candidates") or []):
        if not isinstance(rec, dict) or str(rec.get("kind", "")) != "y_branch":
            continue
        try:
            support = int(rec.get("support_count", 1) or 1)
        except Exception:
            support = 1
        if support < support_min:
            continue
        rec_payload = rec.get("payload") if isinstance(rec.get("payload"), dict) else {}
        name = str(rec_payload.get("y_transform") or "").strip()
        if not name or name == "identity" or name in seen:
            continue
        seen.add(name)
        out.append(name)
    return out


def split_reservoir_path_string(value: Any) -> list[str]:
    if value is None:
        return []
    text = str(value).strip()
    if not text:
        return []
    raw = text.replace(os.pathsep, ",").split(",")
    return [p.strip() for p in raw if p.strip()]


def _extract_reservoir_payload(obj: Any, *, source: str) -> Optional[dict[str, Any]]:
    if not isinstance(obj, dict):
        return None
    payload = None
    stageb = obj.get("stageB")
    if isinstance(stageb, dict):
        payload = stageb.get("coe_proposal_reservoir")
    if payload is None:
        payload = obj.get("coe_proposal_reservoir")
    if payload is None and isinstance(obj.get("candidates"), list):
        payload = obj
    if not isinstance(payload, dict):
        return None
    out = copy.deepcopy(payload)
    out["source"] = str(source)
    return out


def _extract_stageA_proposal_payload(obj: Any, *, source: str) -> Optional[dict[str, Any]]:
    if not isinstance(obj, dict):
        return None
    payload = None
    stagea = obj.get("stageA")
    if isinstance(stagea, dict):
        payload = stagea.get("coe_stageA_proposal_reservoir")
    if payload is None:
        payload = obj.get("coe_stageA_proposal_reservoir")
    if payload is None and str(obj.get("kind", "")) == "stageA_proposal_reservoir":
        payload = obj
    if not isinstance(payload, dict):
        return None
    out = copy.deepcopy(payload)
    out["source"] = str(source)
    return out


def load_proposal_reservoir_payloads(
    paths: Sequence[str | Path],
    *,
    problem_stem: Optional[str] = None,
    max_files: int = 128,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Load proposal reservoirs from report JSON files or result directories."""
    payloads: list[dict[str, Any]] = []
    warnings: list[str] = []
    seen_files: set[str] = set()

    def _candidate_files(path: Path) -> list[Path]:
        if path.is_dir():
            stem = str(problem_stem or "").strip()
            patterns = [f"{stem}.report.json", f"{stem}_*.report.json"] if stem else ["*.report.json"]
            out: list[Path] = []
            for pattern in patterns:
                out.extend(sorted(path.glob(pattern)))
            return out
        return [path]

    for raw in list(paths or []):
        p = Path(str(raw)).expanduser()
        files = _candidate_files(p)
        if not files:
            warnings.append(f"reservoir source has no matching report files: {raw}")
            continue
        for file in files:
            if len(seen_files) >= int(max_files):
                warnings.append(f"reservoir file cap reached ({max_files}); remaining sources skipped")
                return payloads, warnings
            try:
                file_s = str(file.resolve())
            except Exception:
                file_s = str(file)
            if file_s in seen_files:
                continue
            seen_files.add(file_s)
            try:
                with file.open("r", encoding="utf-8") as f:
                    obj = json.load(f)
                payload = _extract_reservoir_payload(obj, source=file_s)
                if payload is not None:
                    payloads.append(payload)
            except Exception as exc:
                warnings.append(f"could not load reservoir source {file}: {type(exc).__name__}: {exc}")
    return payloads, warnings


def load_stageA_proposal_reservoir_payloads(
    paths: Sequence[str | Path],
    *,
    problem_stem: Optional[str] = None,
    max_files: int = 128,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Load Stage-A proposal reservoirs from report JSON files or directories."""
    payloads: list[dict[str, Any]] = []
    warnings: list[str] = []
    seen_files: set[str] = set()

    def _candidate_files(path: Path) -> list[Path]:
        if path.is_dir():
            stem = str(problem_stem or "").strip()
            patterns = [f"{stem}.report.json", f"{stem}_*.report.json"] if stem else ["*.report.json"]
            out: list[Path] = []
            for pattern in patterns:
                out.extend(sorted(path.glob(pattern)))
            return out
        return [path]

    for raw in list(paths or []):
        p = Path(str(raw)).expanduser()
        files = _candidate_files(p)
        if not files:
            warnings.append(f"stageA proposal reservoir source has no matching report files: {raw}")
            continue
        for file in files:
            if len(seen_files) >= int(max_files):
                warnings.append(f"stageA proposal reservoir file cap reached ({max_files}); remaining sources skipped")
                return payloads, warnings
            try:
                file_s = str(file.resolve())
            except Exception:
                file_s = str(file)
            if file_s in seen_files:
                continue
            seen_files.add(file_s)
            try:
                with file.open("r", encoding="utf-8") as f:
                    obj = json.load(f)
                payload = _extract_stageA_proposal_payload(obj, source=file_s)
                if payload is not None:
                    payloads.append(payload)
            except Exception as exc:
                warnings.append(f"could not load Stage-A proposal reservoir source {file}: {type(exc).__name__}: {exc}")
    return payloads, warnings


def _add_reservoir_candidates(
    out: list[CandidateArtifact],
    seen: Optional[set[str]],
    reservoir_payload: Any,
    *,
    max_candidates: Optional[int],
    dropped_log: Optional[list[dict[str, Any]]] = None,
) -> None:
    if not isinstance(reservoir_payload, dict):
        return
    if max_candidates is not None and len(out) >= max_candidates:
        return
    for rec in list(reservoir_payload.get("candidates") or []):
        if max_candidates is not None and len(out) >= max_candidates:
            break
        if not isinstance(rec, dict):
            continue
        meta = _metadata_with_coefficient_bundle(
            dict(rec.get("metadata") or {}),
            rec,
        )
        expr_space = str(meta.get("expression_space") or "raw_y").strip().lower()
        if expr_space not in {"", "raw", "raw_y", "original_y"}:
            continue
        meta.update(
            {
                "reservoir_id": rec.get("reservoir_id"),
                "reservoir_support_count": rec.get("support_count"),
                "reservoir_sources": list(rec.get("sources") or []),
                "reservoir_best_loss": rec.get("best_loss"),
            }
        )
        _add_candidate(
            out,
            seen,
            expr=rec.get("expr"),
            source="proposal_reservoir:" + str(rec.get("source", "unknown")),
            label=rec.get("label") or "proposal_reservoir",
            complexity=rec.get("complexity"),
            n_free_params=max(0, int(rec.get("n_free_params") or 0)),
            metadata=meta,
            dropped_log=dropped_log,
        )


def collect_final_candidates(
    *,
    stageB_data: Optional[dict[str, Any]],
    final_polish_summary: Optional[dict[str, Any]],
    max_candidates: Optional[int] = 16,
    include_reservoir: bool = True,
    deduplicate: bool = True,
    dropped_log: Optional[list[dict[str, Any]]] = None,
) -> list[CandidateArtifact]:
    """Collect visible final analytic expressions for committee scoring.

    ``deduplicate=False`` is intended for provenance-preserving archive export;
    committee scoring retains the historical deduplicated default.

    ``dropped_log``, when provided, records every non-portable expression the
    collector silently ignores (reason, source, preview).  Without it a run
    whose every candidate carries an unresolved NN/leaf atom reports "no
    candidates" with no trace of why, which is how the pb101/pb119
    empty-archive failures stayed undiagnosed.
    """
    out: list[CandidateArtifact] = []
    limit = (
        max(1, int(max_candidates))
        if max_candidates is not None
        else None
    )
    seen: Optional[set[str]] = set() if bool(deduplicate) else None
    stageb_coefficient_metadata = None
    stageb_y_artifacts_added = False
    if isinstance(stageB_data, dict):
        stageb_metadata = _metadata_with_coefficient_bundle({}, stageB_data)
        stageb_coefficient_metadata = stageb_metadata.get("coefficient_metadata")
        if stageb_coefficient_metadata is None:
            stageb_sympy_meta = stageB_data.get("sympy_meta")
            if isinstance(stageb_sympy_meta, dict):
                stageb_coefficient_metadata = stageb_sympy_meta.get(
                    "coefficient_metadata"
                )

    if isinstance(final_polish_summary, dict):
        polish_coefficient_metadata = final_polish_summary.get(
            "coefficient_metadata"
        )
        rec = final_polish_summary.get("recommended")
        if isinstance(rec, dict):
            rec_metadata = _metadata_with_coefficient_bundle(
                {"is_recommended": True},
                rec,
                fallback=polish_coefficient_metadata,
            )
            rec_unit_certificate = rec.get("unit_admissibility")
            if not isinstance(rec_unit_certificate, dict):
                rec_unit_certificate = final_polish_summary.get(
                    "unit_admissibility"
                )
            if isinstance(rec_unit_certificate, dict):
                rec_metadata["unit_admissibility"] = dict(rec_unit_certificate)
            _add_candidate(
                out,
                seen,
                expr=rec.get("expr"),
                source="final_polish:recommended",
                label=rec.get("label") or "recommended",
                complexity=rec.get("complexity"),
                n_free_params=_selection_n_free_params(rec),
                metadata=rec_metadata,
                dropped_log=dropped_log,
            )
        seed_expr = final_polish_summary.get("seed_expr")
        if seed_expr is not None:
            seed_metadata = _metadata_with_coefficient_bundle(
                {"is_seed": True},
                final_polish_summary,
                fallback=polish_coefficient_metadata,
            )
            seed_unit_certificate = final_polish_summary.get(
                "seed_unit_admissibility"
            )
            if isinstance(seed_unit_certificate, dict):
                seed_metadata["unit_admissibility"] = dict(seed_unit_certificate)
            _add_candidate(
                out,
                seen,
                expr=seed_expr,
                source="final_polish:seed",
                label="seed",
                metadata=seed_metadata,
                dropped_log=dropped_log,
            )
        if isinstance(stageB_data, dict):
            for rec in list(stageB_data.get("y_branch_artifacts") or []):
                if limit is not None and len(out) >= limit:
                    break
                if not isinstance(rec, dict):
                    continue
                meta = _metadata_with_coefficient_bundle(
                    dict(rec.get("metadata") or {}),
                    rec,
                )
                meta.setdefault("stageB_y_branch_artifact", True)
                _add_candidate(
                    out,
                    seen,
                    expr=rec.get("expr"),
                    source=rec.get("source") or "stageB_y_branch",
                    label=rec.get("label") or "stageB_y_branch",
                    complexity=rec.get("complexity"),
                    n_free_params=int(rec.get("n_free_params") or 0),
                    metadata=meta,
                    dropped_log=dropped_log,
                )
            stageb_y_artifacts_added = True
        for group_name in ("all_candidates", "strict_pareto", "epsilon_pareto"):
            for rec_i in list(final_polish_summary.get(group_name) or []):
                if limit is not None and len(out) >= limit:
                    break
                if not isinstance(rec_i, dict):
                    continue
                pareto_metadata = _metadata_with_coefficient_bundle(
                    {"pareto_group": group_name},
                    rec_i,
                    fallback=polish_coefficient_metadata,
                )
                pareto_unit_certificate = rec_i.get("unit_admissibility")
                if isinstance(pareto_unit_certificate, dict):
                    pareto_metadata["unit_admissibility"] = dict(
                        pareto_unit_certificate
                    )
                _add_candidate(
                    out,
                    seen,
                    expr=rec_i.get("expr"),
                    source=f"final_polish:{group_name}",
                    label=rec_i.get("label") or group_name,
                    complexity=rec_i.get("complexity"),
                    n_free_params=_selection_n_free_params(rec_i),
                    metadata=pareto_metadata,
                    dropped_log=dropped_log,
                )

    if isinstance(stageB_data, dict):
        if not stageb_y_artifacts_added:
            for rec in list(stageB_data.get("y_branch_artifacts") or []):
                if limit is not None and len(out) >= limit:
                    break
                if not isinstance(rec, dict):
                    continue
                meta = _metadata_with_coefficient_bundle(
                    dict(rec.get("metadata") or {}),
                    rec,
                )
                meta.setdefault("stageB_y_branch_artifact", True)
                _add_candidate(
                    out,
                    seen,
                    expr=rec.get("expr"),
                    source=rec.get("source") or "stageB_y_branch",
                    label=rec.get("label") or "stageB_y_branch",
                    complexity=rec.get("complexity"),
                    n_free_params=int(rec.get("n_free_params") or 0),
                    metadata=meta,
                    dropped_log=dropped_log,
                )
        sympy_meta = stageB_data.get("sympy_meta") or {}
        stagec_ok = _stagec_expression_certified(sympy_meta)
        if stagec_ok:
            stagec_metadata = _metadata_with_coefficient_bundle(
                {"stagec": True, "stagec_certified": True},
                sympy_meta if isinstance(sympy_meta, dict) else None,
                fallback=stageb_coefficient_metadata,
            )
            unit_admissibility = (
                sympy_meta.get("unit_admissibility")
                if isinstance(sympy_meta, dict)
                else None
            )
            if isinstance(unit_admissibility, dict):
                stagec_metadata["unit_admissibility"] = dict(unit_admissibility)
            _add_candidate(
                out,
                seen,
                expr=stageB_data.get("y_expr_raw_str") or stageB_data.get("y_expr_str"),
                source="stageC:y_expr",
                label="stageC",
                metadata=stagec_metadata,
                dropped_log=dropped_log,
            )
        else:
            stagec_expr = stageB_data.get("y_expr_raw_str") or stageB_data.get("y_expr_str")
            _append_expression_drop(
                dropped_log,
                expr=stagec_expr,
                reason=_uncertified_stagec_drop_reason(stagec_expr, sympy_meta),
                source="stageC:y_expr",
                label="stageC",
            )
        if bool(include_reservoir):
            _add_reservoir_candidates(
                out,
                seen,
                stageB_data.get("coe_proposal_reservoir"),
                max_candidates=limit,
                dropped_log=dropped_log,
            )

    # Fill missing complexity after dedupe.  Keep this best-effort because
    # committee scoring must not fail because SymPy dislikes one candidate.
    fixed: list[CandidateArtifact] = []
    selected = out if limit is None else out[:limit]
    for cand in selected:
        if cand.complexity is not None:
            fixed.append(cand)
            continue
        try:
            cx = float(expression_complexity(cand.expr, PolishConfig(), n_free_params=cand.n_free_params))
        except Exception:
            cx = float(len(str(cand.expr)))
        fixed.append(
            CandidateArtifact(
                candidate_id=cand.candidate_id,
                expr=cand.expr,
                source=cand.source,
                label=cand.label,
                complexity=cx,
                n_free_params=cand.n_free_params,
                metadata=dict(cand.metadata),
            )
        )
    return fixed


def build_slice_specs(
    *,
    n_slices: int,
    ndata_train: int,
    ndata_val: int,
    start_slice: int = 0,
    skip_slice_ids: Optional[Sequence[int]] = None,
    max_rows: Optional[int] = None,
) -> list[SliceSpec]:
    specs: list[SliceSpec] = []
    n_train = int(ndata_train)
    n_val = int(ndata_val)
    block = n_train + n_val
    skip_ids: set[int] = set()
    for raw in list(skip_slice_ids or []):
        try:
            skip_ids.add(int(raw))
        except Exception:
            continue
    sid = int(start_slice)
    while len(specs) < max(0, int(n_slices)):
        if int(sid) in skip_ids:
            sid += 1
            continue
        start = int(sid) * block
        stop = start + block
        if max_rows is not None:
            try:
                if stop > int(max_rows):
                    break
            except Exception:
                pass
        specs.append(
            SliceSpec(
                slice_id=int(sid),
                train_start=start,
                train_stop=start + n_train,
                val_start=start + n_train,
                val_stop=stop,
            )
        )
        sid += 1
    return specs


@lru_cache(maxsize=8)
def _load_dataset_arrays(filepath_s: str) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    if pd is None:
        raise RuntimeError("pandas is required for CoE committee slice loading")
    df = pd.read_csv(filepath_s)
    if "y" not in df.columns:
        raise ValueError("target column 'y' not found")
    var_cols = [c for c in df.columns if c != "y"]
    var_cols.sort(key=lambda s: (0, int(s[1:])) if s.startswith("x") and s[1:].isdigit() else (1, s))
    X = df[var_cols].to_numpy(dtype=np.float64)
    y = df["y"].to_numpy(dtype=np.float64).reshape(-1)
    return X, y, tuple(var_cols)


def _load_val_slice(filepath: str | Path, spec: SliceSpec) -> tuple[np.ndarray, np.ndarray, list[str]]:
    filepath_s = str(Path(filepath).expanduser().resolve())
    X_all, y_all, names_t = _load_dataset_arrays(filepath_s)
    if spec.val_stop > int(y_all.size):
        raise ValueError(
            f"slice {spec.slice_id} requires rows up to {spec.val_stop}, but dataset has {int(y_all.size)}"
        )
    start = int(spec.val_start)
    stop = int(spec.val_stop)
    return X_all[start:stop], y_all[start:stop], list(names_t)


def evaluate_candidate_on_slice(
    cand: CandidateArtifact,
    *,
    filepath: str | Path,
    spec: SliceSpec,
    min_valid_fraction: float = 0.80,
    return_row_losses: bool = False,
) -> SliceFitResult:
    try:
        X, y, names = _load_val_slice(filepath, spec)
        expr_names = infer_variable_names(cand.expr, X)
        if len(expr_names) > len(names):
            names = [f"x{i}" for i in range(len(expr_names))]
        parsed = parse_sympy_expr(cand.expr, names)
        symbol_values = coefficient_symbol_values_for_expression(
            _candidate_coefficient_metadata(cand),
            parsed,
            variable_names=names,
        )
        pred = _eval_expr_array(parsed, X, names, symbol_values=symbol_values)
        mse, se, frac = _mse_and_se(pred, y, min_valid_fraction)
        row_losses: Optional[tuple[float, ...]] = None
        if return_row_losses:
            pred_arr = np.asarray(pred, dtype=np.float64).reshape(-1)
            y_arr = np.asarray(y, dtype=np.float64).reshape(-1)
            n_rows = min(pred_arr.size, y_arr.size)
            rows = np.full(int(y_arr.size), np.nan, dtype=np.float64)
            valid = np.isfinite(pred_arr[:n_rows]) & np.isfinite(y_arr[:n_rows])
            diff = pred_arr[:n_rows] - y_arr[:n_rows]
            rows[:n_rows][valid] = diff[valid] ** 2
            row_losses = tuple(float(v) for v in rows)
        status = "success" if math.isfinite(mse) else "invalid"
        return SliceFitResult(
            candidate_id=cand.candidate_id,
            slice_id=int(spec.slice_id),
            val_mse=float(mse),
            val_mse_se=float(se),
            frac_valid=float(frac),
            n_val=int(y.size),
            status=status,
            error=None if status == "success" else "non-finite or insufficient valid predictions",
            row_losses=row_losses,
        )
    except Exception as exc:
        return SliceFitResult(
            candidate_id=cand.candidate_id,
            slice_id=int(spec.slice_id),
            val_mse=float("inf"),
            val_mse_se=float("inf"),
            frac_valid=0.0,
            n_val=0,
            status="error",
            error=f"{type(exc).__name__}: {exc}",
        )


def _nanmedian(values: Iterable[float]) -> float:
    arr = np.asarray([float(v) for v in values if math.isfinite(float(v))], dtype=np.float64)
    if arr.size == 0:
        return float("inf")
    return float(np.median(arr))


def _committee_tolerance(
    *,
    loss_a: float,
    loss_b: float,
    noise_floor_raw: float,
    n_eff: int,
    noise_mult: float,
    rel_tol: float,
) -> float:
    nf = float(noise_floor_raw) if math.isfinite(float(noise_floor_raw)) else 0.0
    n = max(1, int(n_eff))
    stat = float(noise_mult) * max(nf, 0.0) * math.sqrt(2.0 / float(n))
    rel = float(rel_tol) * max(abs(float(loss_a)), abs(float(loss_b)), nf, 1.0e-30)
    return max(1.0e-30, stat, rel)


def _candidate_support_count(cand_payload: Any) -> int:
    if not isinstance(cand_payload, dict):
        return 1
    meta = cand_payload.get("metadata") or {}
    vals = [
        meta.get("reservoir_support_count") if isinstance(meta, dict) else None,
        cand_payload.get("support_count"),
    ]
    best = 1
    for val in vals:
        try:
            best = max(best, int(val))
        except Exception:
            continue
    return max(1, best)


def _support_adjusted_complexity(
    *,
    complexity: Any,
    support_count: int,
    support_bonus: float,
) -> float:
    try:
        base = float(complexity)
    except Exception:
        base = float("inf")
    if not math.isfinite(base):
        return float("inf")
    bonus = max(0.0, float(support_bonus)) * math.log2(max(1, int(support_count)))
    return float(base - bonus)


def run_final_committee_audit(
    *,
    filepath: str | Path,
    stageB_data: Optional[dict[str, Any]],
    final_polish_summary: Optional[dict[str, Any]],
    mode: str,
    n_slices: int,
    start_slice: int,
    ndata_train: int,
    ndata_val: int,
    max_candidates: int = 16,
    noise_floor_raw: float = 0.0,
    noise_mult: float = 3.0,
    rel_tol: float = 1.0e-3,
    min_valid_fraction: float = 0.80,
    reservoir_support_bonus: float = 0.0,
    cache: Optional[CommitteeEvalCache] = None,
    reference_slice: Optional[int] = None,
    include_reservoir: bool = False,
    witness_parallelism: int = 1,
) -> CommitteeDecision:
    excluded_slice_ids: tuple[int, ...] = ()
    if reference_slice is not None:
        try:
            excluded_slice_ids = (int(reference_slice),)
        except Exception:
            excluded_slice_ids = ()
    candidates = collect_final_candidates(
        stageB_data=stageB_data,
        final_polish_summary=final_polish_summary,
        max_candidates=max_candidates,
        include_reservoir=bool(include_reservoir),
    )
    max_rows: Optional[int] = None
    try:
        _X_all, _y_all, _names = _load_dataset_arrays(str(Path(filepath).expanduser().resolve()))
        max_rows = int(_y_all.size)
    except Exception:
        max_rows = None
    specs = build_slice_specs(
        n_slices=max(0, int(n_slices)),
        ndata_train=max(1, int(ndata_train)),
        ndata_val=max(1, int(ndata_val)),
        start_slice=max(0, int(start_slice)),
        skip_slice_ids=excluded_slice_ids,
        max_rows=max_rows,
    )
    cfg = {
        "mode": mode,
        "n_slices": int(n_slices),
        "start_slice": int(start_slice),
        "ndata_train": int(ndata_train),
        "ndata_val": int(ndata_val),
        "max_candidates": int(max_candidates),
        "noise_floor_raw": float(noise_floor_raw),
        "noise_mult": float(noise_mult),
        "rel_tol": float(rel_tol),
        "min_valid_fraction": float(min_valid_fraction),
        "reservoir_support_bonus": float(reservoir_support_bonus),
        "reference_slice": None if reference_slice is None else int(reference_slice),
        "excluded_slice_ids": list(excluded_slice_ids),
        "include_reservoir": bool(include_reservoir),
        "witness_parallelism": max(1, int(witness_parallelism or 1)),
    }
    warnings: list[str] = []
    if not candidates:
        return CommitteeDecision(
            mode=mode,
            status="skipped",
            incumbent_id=None,
            recommended_id=None,
            recommended_expr=None,
            n_candidates=0,
            n_slices=len(specs),
            candidate_summary=[],
            slice_specs=[s.to_dict() for s in specs],
            results=[],
            config=cfg,
            warnings=["no analytic final candidates"],
        )

    results: list[SliceFitResult] = []
    witness_executor_meta = None
    if max(1, int(witness_parallelism or 1)) > 1:
        try:
            from nestynet_sr.sr_search.coe_witness import (
                CoEWitnessExecutor,
                coe_witness_execution_metadata,
                run_fixed_expression_candidate_witnesses,
            )

            witness_executor = CoEWitnessExecutor(parallelism=max(1, int(witness_parallelism or 1)))
            rows = run_fixed_expression_candidate_witnesses(
                specs=specs,
                candidates=candidates,
                filepath=str(filepath),
                min_valid_fraction=float(min_valid_fraction),
                executor=witness_executor,
                prefix="final_coe",
            )
            witness_executor_meta = coe_witness_execution_metadata(witness_executor, rows)
            for row in rows:
                payload = {
                    key: row.get(key)
                    for key in (
                        "candidate_id",
                        "slice_id",
                        "val_mse",
                        "val_mse_se",
                        "frac_valid",
                        "n_val",
                        "status",
                        "error",
                    )
                }
                results.append(SliceFitResult(**payload))
        except Exception as exc:
            warnings.append(
                f"parallel final committee witness evaluation failed; using serial: "
                f"{type(exc).__name__}: {exc}"
            )
            results = []
    if not results:
        for spec in specs:
            for cand in candidates:
                results.append(
                    evaluate_candidate_on_slice_cached(
                        cand,
                        filepath=filepath,
                        spec=spec,
                        min_valid_fraction=min_valid_fraction,
                        cache=cache,
                    )
                )
        if witness_executor_meta is None:
            witness_executor_meta = {
                "backend": "serial",
                "parallelism": max(1, int(witness_parallelism or 1)),
                "effective_backend": "serial",
            }
    cfg["witness_executor"] = witness_executor_meta

    by_cid: dict[str, list[SliceFitResult]] = {cand.candidate_id: [] for cand in candidates}
    for res in results:
        by_cid.setdefault(res.candidate_id, []).append(res)

    incumbent = candidates[0]
    incumbent_losses = [r.val_mse for r in by_cid.get(incumbent.candidate_id, []) if r.status == "success"]
    incumbent_median = _nanmedian(incumbent_losses)

    summary: list[dict[str, Any]] = []
    for cand in candidates:
        rows = by_cid.get(cand.candidate_id, [])
        ok_rows = [r for r in rows if r.status == "success"]
        losses = [r.val_mse for r in ok_rows]
        med = _nanmedian(losses)
        mean = float(np.mean(losses)) if losses else float("inf")
        wins = losses_ct = ties = 0
        if cand.candidate_id != incumbent.candidate_id:
            inc_by_slice = {r.slice_id: r for r in by_cid.get(incumbent.candidate_id, [])}
            for r in ok_rows:
                inc = inc_by_slice.get(r.slice_id)
                if inc is None or inc.status != "success":
                    continue
                tol = _committee_tolerance(
                    loss_a=inc.val_mse,
                    loss_b=r.val_mse,
                    noise_floor_raw=noise_floor_raw,
                    n_eff=max(1, r.n_val),
                    noise_mult=noise_mult,
                    rel_tol=rel_tol,
                )
                delta = r.val_mse - inc.val_mse
                if delta < -tol:
                    wins += 1
                elif delta > tol:
                    losses_ct += 1
                else:
                    ties += 1
        support_count = _candidate_support_count(cand.to_dict())
        complexity_raw = cand.complexity
        complexity_support_adjusted = _support_adjusted_complexity(
            complexity=complexity_raw,
            support_count=support_count,
            support_bonus=reservoir_support_bonus,
        )
        summary.append(
            {
                "candidate": cand.to_dict(),
                "status": "success" if ok_rows else "invalid",
                "n_success": len(ok_rows),
                "n_error": len(rows) - len(ok_rows),
                "median_val_mse": med,
                "mean_val_mse": mean,
                "reservoir_support_count": int(support_count),
                "support_adjusted_complexity": float(complexity_support_adjusted),
                "delta_median_vs_incumbent": med - incumbent_median if math.isfinite(incumbent_median) else float("nan"),
                "wins_vs_incumbent": wins,
                "ties_vs_incumbent": ties,
                "losses_vs_incumbent": losses_ct,
            }
        )

    valid_summary = [row for row in summary if row.get("status") == "success"]

    def loss_rank_key(row: dict[str, Any]) -> tuple[float, float, str]:
        cand = row.get("candidate") or {}
        return (
            float(row.get("median_val_mse", float("inf"))),
            float(cand.get("complexity", float("inf")) or float("inf")),
            str(cand.get("candidate_id", "")),
        )

    loss_ranked = sorted(valid_summary, key=loss_rank_key)
    best_loss_row = loss_ranked[0] if loss_ranked else None
    if best_loss_row is not None:
        best_loss = float(best_loss_row.get("median_val_mse", float("inf")))
        best_success = max(1, int(best_loss_row.get("n_success", 0) or 0))
        n_eff = max(1, int(ndata_val) * best_success)
        tied_rows = []
        for row in valid_summary:
            med = float(row.get("median_val_mse", float("inf")))
            tol = _committee_tolerance(
                loss_a=best_loss,
                loss_b=med,
                noise_floor_raw=noise_floor_raw,
                n_eff=n_eff,
                noise_mult=noise_mult,
                rel_tol=rel_tol,
            )
            if med <= best_loss + tol:
                row["noise_tied_with_best"] = True
                row["noise_tie_tolerance_vs_best"] = float(tol)
                tied_rows.append(row)
            else:
                row["noise_tied_with_best"] = False
                row["noise_tie_tolerance_vs_best"] = float(tol)

        use_exactness_prior = bool(float(noise_floor_raw) > 0.0 or float(best_loss) <= 1.0e-24)

        def adjudication_rank_key(row: dict[str, Any]) -> tuple[float, float, float, str]:
            cand = row.get("candidate") or {}
            support_count = int(row.get("reservoir_support_count", 1) or 1)
            base_complexity = float(
                row.get("support_adjusted_complexity", cand.get("complexity", float("inf")))
                or float("inf")
            )
            exactness_penalty = _symbolic_exactness_adjustment(cand) if use_exactness_prior else 0.0
            row["symbolic_exactness_adjustment"] = float(exactness_penalty)
            row["adjudication_complexity"] = float(base_complexity + exactness_penalty)
            return (
                float(base_complexity + exactness_penalty),
                -float(support_count),
                float(row.get("median_val_mse", float("inf"))),
                str(cand.get("candidate_id", "")),
            )

        adjudication_pool = tied_rows or loss_ranked
        recommended_row = sorted(adjudication_pool, key=adjudication_rank_key)[0]
        if tied_rows and recommended_row is not best_loss_row:
            rec_support = int(recommended_row.get("reservoir_support_count", 1) or 1)
            selection_basis = (
                "noise_tied_support_adjusted_complexity"
                if float(reservoir_support_bonus) > 0.0 and rec_support > 1
                else "noise_tied_complexity"
            )
        else:
            selection_basis = "median_loss"
    else:
        recommended_row = None
        selection_basis = "none"
    recommended_id = None
    recommended_expr = None
    status = "success"
    if recommended_row is not None:
        cand_payload = recommended_row.get("candidate") or {}
        recommended_id = cand_payload.get("candidate_id")
        recommended_expr = cand_payload.get("expr")
        if recommended_id != incumbent.candidate_id:
            if str(mode) in {"final_adjudicate", "committee_gated", "reservoir_discovery"}:
                warnings.append("committee selected a different final candidate; CoE final selection applies it")
            else:
                warnings.append(
                    "committee selected a different final candidate; audit_final reports this but does not alter final selection"
                )
    else:
        status = "skipped"

    cfg["selection_basis"] = selection_basis
    if cache is not None:
        cfg["cache_stats"] = cache.stats()

    return CommitteeDecision(
        mode=mode,
        status=status,
        incumbent_id=incumbent.candidate_id,
        recommended_id=recommended_id,
        recommended_expr=recommended_expr,
        n_candidates=len(candidates),
        n_slices=len(specs),
        candidate_summary=summary,
        slice_specs=[s.to_dict() for s in specs],
        results=[r.to_dict() for r in results],
        config=cfg,
        warnings=warnings,
    )


def write_committee_jsonl(path: str | Path, decision: CommitteeDecision) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for row in decision.results:
            f.write(json.dumps(row, sort_keys=True) + "\n")
