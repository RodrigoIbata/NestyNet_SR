# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""Ordinary-SR adapter for frozen, confidence-bearing Pareto selection.

The search engines remain free to use validation losses and physics-aware
heuristics to allocate computation.  This module gives those engines a clean
statistical boundary:

* reserve an audit data view before search starts;
* freeze a canonical union of portable symbolic candidates;
* evaluate every candidate on the same untouched units and domain;
* construct the simultaneous confidence Pareto graph from paired losses.

No audit response is returned by :func:`prepare_sr_audit_plan`; search receives
only the physical search-view path.  Apart from pre-search schema validation,
splitting, and cryptographic sealing, audit values are not evaluated until the
candidate archive has been frozen.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Optional

import numpy as np

from .archive import CandidateArchive
from .audit import AuditDesign, LossAudit
from .calibration_profile import select_inference_method
from .certificate import build_certificate
from .complexity import ComplexityVector
from .pareto import (
    ConfidenceParetoResult,
    _bonferroni_t_critical_value,
    _max_t_critical_value,
    bootstrap_front_inclusion_frequencies,
    confidence_pareto,
)


class NoPortableAnalyticCandidatesError(ValueError):
    """The search completed without a candidate eligible for certification."""


@dataclass(frozen=True)
class SRAuditPlan:
    """Physical search/audit split established before ordinary SR begins."""

    source_path: str
    search_path: str
    audit_path: str
    source_sha256: str
    search_sha256: str
    audit_sha256: str
    source_rows: int
    search_rows: int
    audit_rows: int
    audit_kind: str
    audit_start_row: Optional[int] = None
    external_audit_source: Optional[str] = None
    target_column: str = "y"
    unit_size: int = 1
    schema_version: int = 2

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def fingerprint(self) -> str:
        """Fingerprint the complete provenance record, including local paths."""

        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @property
    def contract_fingerprint(self) -> str:
        """Path-independent identity of the sealed search/audit data contract."""

        return str(self.checkpoint_contract()["contract_fingerprint"])

    def checkpoint_contract(self) -> dict[str, Any]:
        """Return the path-independent split identity stored in SR checkpoints."""

        contract = {
            "schema_version": int(self.schema_version),
            "source_sha256": self.source_sha256,
            "search_sha256": self.search_sha256,
            "audit_sha256": self.audit_sha256,
            "source_rows": int(self.source_rows),
            "search_rows": int(self.search_rows),
            "audit_rows": int(self.audit_rows),
            "audit_kind": self.audit_kind,
            "audit_start_row": self.audit_start_row,
            "target_column": self.target_column,
            "unit_size": int(self.unit_size),
        }
        encoded = json.dumps(contract, sort_keys=True, separators=(",", ":"))
        contract["contract_fingerprint"] = hashlib.sha256(
            encoded.encode("utf-8")
        ).hexdigest()
        return contract

    def assert_checkpoint_compatible(self, payload: Mapping[str, Any]) -> None:
        """Reject a resume state trained behind a different audit firewall."""

        if not isinstance(payload, Mapping):
            raise ValueError(
                "checkpoint predates the statistical audit firewall; restart search "
                "or resume without requesting statistical certification"
            )
        expected = self.checkpoint_contract()
        mismatches = {
            key: {"expected": value, "observed": payload.get(key)}
            for key, value in expected.items()
            if payload.get(key) != value
        }
        if mismatches:
            raise ValueError(
                "checkpoint statistical split does not match the current audit "
                f"firewall: {mismatches!r}"
            )


@dataclass(frozen=True)
class SRArchiveBuild:
    """Frozen ordinary-SR archive plus transparent collection diagnostics."""

    archive: CandidateArchive
    discovered_count: int
    canonical_count: int
    excluded: tuple[Mapping[str, Any], ...]
    cap_applied: bool
    cap_policy: str
    upstream_drops: tuple[Mapping[str, Any], ...] = ()

    def upstream_drop_reasons(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for row in self.upstream_drops:
            reason = str(row.get("reason", "unknown"))
            counts[reason] = counts.get(reason, 0) + 1
        return counts

    def to_dict(self) -> dict[str, Any]:
        return {
            "discovered_count": int(self.discovered_count),
            "canonical_count": int(self.canonical_count),
            "archived_count": len(self.archive),
            "excluded": [dict(item) for item in self.excluded],
            "cap_applied": bool(self.cap_applied),
            "cap_policy": self.cap_policy,
            "archive_fingerprint": self.archive.fingerprint,
            "upstream_drops": {
                "count": len(self.upstream_drops),
                "reasons": self.upstream_drop_reasons(),
                "samples": [dict(item) for item in self.upstream_drops[:8]],
            },
        }


@dataclass(frozen=True)
class SRAuditEvaluation:
    """Common-domain loss audit and evaluation diagnostics."""

    audit: LossAudit
    variable_names: tuple[str, ...]
    scale: float
    scale_name: str
    unit_size: int
    eligible_candidate_ids: tuple[str, ...]
    candidate_failures: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    # Per-candidate audit predictions and the row -> unit map, kept in memory
    # for functional-equivalence testing.  Deliberately absent from to_dict:
    # they are working arrays, not certificate content.
    predictions: Mapping[str, Any] = field(default_factory=dict)
    row_unit_index: Any = None
    # Standardized loss of the best constant model on the audit rows.  This is
    # the "say nothing" baseline the compression test measures against.
    null_total_standardized_loss: float = 0.0
    n_audit_rows: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "audit_fingerprint": self.audit.fingerprint,
            "n_rows": int(sum(int(m.get("n_rows", 0)) for m in self.audit.unit_metadata)),
            "n_units": self.audit.n_units,
            "variable_names": list(self.variable_names),
            "scale": float(self.scale),
            "scale_name": self.scale_name,
            "unit_size": int(self.unit_size),
            "eligible_candidate_ids": list(self.eligible_candidate_ids),
            "ineligible_candidate_ids": [
                candidate_id
                for candidate_id in self.audit.candidate_ids
                if candidate_id not in set(self.eligible_candidate_ids)
            ],
            "failure_loss": self.audit.failure_loss,
            "candidate_failures": {
                str(key): dict(value) for key, value in self.candidate_failures.items()
            },
        }


@dataclass(frozen=True)
class DeploymentNoninferiorityResult:
    """Simultaneous risk-only set retained as a deployment firewall.

    The Pareto fronts remain the scientific output.  This auxiliary set uses
    an all-candidate, one-sided multiplier max-range bound, so its empirical
    risk reference may be selected after seeing the audit without silently
    dropping the winner-selection multiplicity.
    """

    candidate_ids: tuple[str, ...]
    reference_candidate_id: str
    noninferiority_set: tuple[str, ...]
    risk_differences: tuple[float, ...]
    standard_errors: tuple[float, ...]
    upper_confidence_bounds: tuple[float, ...]
    estimable: tuple[bool, ...]
    eligible: tuple[bool, ...]
    alpha: float
    delta: float
    critical_radius: float
    n_resamples: int
    seed: int
    multiplier: str
    audit_fingerprint: str
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        def finite_or_none(value: float) -> Optional[float]:
            number = float(value)
            return number if math.isfinite(number) else None

        comparisons = {}
        for index, candidate_id in enumerate(self.candidate_ids):
            comparisons[candidate_id] = {
                "risk_difference_to_reference": float(self.risk_differences[index]),
                "standard_error": float(self.standard_errors[index]),
                "upper_confidence_bound": finite_or_none(
                    self.upper_confidence_bounds[index]
                ),
                "estimable": bool(self.estimable[index]),
                "eligible": bool(self.eligible[index]),
                "included": candidate_id in self.noninferiority_set,
            }
        return {
            "method": "all_eligible_candidate_multiplier_max_range",
            "role": "secondary_deployment_firewall",
            "claim": (
                "Every included non-reference candidate has a simultaneous "
                "one-sided upper bound no larger than delta for its risk "
                "difference from the audit-risk minimizer; the reference is "
                "included by identity."
            ),
            "note": "This screen does not choose the structural-identification winner.",
            "reference_candidate_id": self.reference_candidate_id,
            "noninferiority_set": list(self.noninferiority_set),
            "alpha": float(self.alpha),
            "delta": float(self.delta),
            "critical_radius": float(self.critical_radius),
            "n_resamples": int(self.n_resamples),
            "seed": int(self.seed),
            "multiplier": self.multiplier,
            "audit_fingerprint": self.audit_fingerprint,
            "comparisons": comparisons,
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class IdentificationSelectionResult:
    """Occam decision: added complexity must earn a certified improvement."""

    candidate_ids: tuple[str, ...]
    complexity_order: tuple[str, ...]
    complexity_order_records: tuple[Mapping[str, Any], ...]
    selected_candidate_id: str
    challenges: tuple[Mapping[str, Any], ...]
    alpha: float
    delta: float
    numerical_tie_multiplier: float
    critical_value: float
    critical_value_method: str
    n_resamples: int
    seed: int
    multiplier: str
    comparison_family_size_pre_audit: int
    inference_regime: Mapping[str, Any]
    audit_fingerprint: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": "predeclared_complexity_order_challenge_walk",
            "role": "authoritative_identification_decision",
            "claim": "Added complexity must have a simultaneous studentised lower risk-improvement bound larger than delta.",
            "selection_basis": (
                "simplicity_default_complexity_requires_simultaneous_evidence"
            ),
            "candidate_ids": list(self.candidate_ids),
            "complexity_order": list(self.complexity_order),
            "complexity_order_rule": "lexicographic(free_parameters,constant_code,ast_nodes,tree_depth,candidate_id);missing=+infinity",
            "complexity_order_records": [dict(item) for item in self.complexity_order_records],
            "selected_candidate_id": self.selected_candidate_id,
            "challenges": [dict(item) for item in self.challenges],
            "alpha": float(self.alpha),
            "delta": float(self.delta),
            "numerical_tie_rule": (
                "max_abs_paired_unit_loss_difference <= max((multiplier*eps)^2, "
                "multiplier*eps*max_abs_paired_losses)"
            ),
            "numerical_tie_multiplier": float(self.numerical_tie_multiplier),
            "critical_value": float(self.critical_value),
            "critical_value_method": self.critical_value_method,
            "n_resamples": int(self.n_resamples),
            "seed": int(self.seed),
            "multiplier": self.multiplier,
            "comparison_family_size_pre_audit": int(self.comparison_family_size_pre_audit),
            "inference_regime": dict(self.inference_regime),
            "audit_fingerprint": self.audit_fingerprint,
        }


@dataclass(frozen=True)
class SRSelectionOutcome:
    """Files and report payload produced by the ordinary-SR audit."""

    archive_build: SRArchiveBuild
    evaluation: SRAuditEvaluation
    pareto: ConfidenceParetoResult
    deployment: DeploymentNoninferiorityResult
    identification: IdentificationSelectionResult
    selected_candidate_id: str
    front_inclusion_frequencies: Mapping[str, float]
    archive_path: str
    certificate_path: str
    # Serialised functional classes, so a candidate can be mapped to the class
    # whose risk and front membership it inherits.
    functional_classes: Sequence[Mapping[str, Any]] = ()

    def summary(self) -> dict[str, Any]:
        archive = self.archive_build.archive
        risk_by_id = self.pareto.risk_by_id()
        selected = archive[self.selected_candidate_id]
        candidates = {}
        eligible_ids = set(self.pareto.eligible_candidate_ids)
        # Risks and front membership are now indexed by functional class, so a
        # candidate inherits them through the class it belongs to.  Several
        # spellings of one function therefore report the same risk, which is
        # the point: they are one hypothesis.
        class_of: dict[str, str] = {}
        for entry in self.functional_classes:
            for member in entry.get("members", ()):
                class_of[str(member)] = str(entry.get("class_id"))
        for candidate in archive.candidates:
            class_id = class_of.get(candidate.candidate_id)
            candidates[candidate.candidate_id] = {
                "expression": candidate.metadata.get("expression"),
                "complexity": candidate.complexity.as_dict(),
                "functional_class": class_id,
                "risk": risk_by_id.get(class_id) if class_id else None,
                "eligible_for_inference": bool(class_id in eligible_ids),
                "front_inclusion_frequency": float(
                    self.front_inclusion_frequencies.get(class_id, 0.0)
                    if class_id else 0.0
                ),
                "provenance": _json_safe(candidate.provenance),
            }
        return {
            "enabled": True,
            "status": "certified",
            "authority": "statistical_selection",
            "selection_basis": (
                "simplicity_default_complexity_requires_simultaneous_evidence"
            ),
            "selected_candidate_id": self.selected_candidate_id,
            "selected_expression": selected.metadata.get("expression"),
            "selected_coefficient_metadata": _json_safe(
                selected.metadata.get("coefficient_metadata")
            ),
            "selected_unit_admissibility": _json_safe(
                selected.metadata.get("unit_admissibility")
            ),
            "selected_complexity": selected.complexity.as_dict(),
            "selected_functional_class": class_of.get(self.selected_candidate_id),
            "selected_risk": risk_by_id.get(
                class_of.get(self.selected_candidate_id, "")
            ),
            "point_front": list(self.pareto.point_front),
            "confidence_front": list(self.pareto.confidence_front),
            "practical_front": list(self.pareto.practical_front),
            "identification_selection": self.identification.to_dict(),
            "deployment_noninferiority": self.deployment.to_dict(),
            "alpha": float(self.pareto.alpha),
            "delta": float(self.pareto.delta),
            "critical_value": float(self.pareto.critical_value),
            "n_candidates": len(archive),
            "n_units": self.evaluation.audit.n_units,
            "archive_path": self.archive_path,
            "certificate_path": self.certificate_path,
            "archive": self.archive_build.to_dict(),
            "audit": self.evaluation.to_dict(),
            "pareto": self.pareto.to_dict(include_comparisons=False),
            "candidates": candidates,
            "warnings": list(self.pareto.warnings),
        }


def prepare_sr_audit_plan(
    source_path: str | Path,
    *,
    results_dir: str | Path,
    external_audit_path: str | Path | None = None,
    audit_rows: int = 0,
    audit_fraction: float = 0.2,
    minimum_search_rows: int = 2,
    minimum_audit_rows: int = 2,
    unit_size: int = 1,
    target_column: str | None = None,
) -> SRAuditPlan:
    """Create immutable CSV views before any search code opens the data.

    With an external audit file, search uses the complete source file.  Without
    one, the final contiguous tail is reserved and search receives a new CSV
    containing only the prefix.  View filenames include content hashes, so an
    old checkpoint cannot silently resume behind a different audit boundary.
    """

    pd = _require_pandas()
    unit = int(unit_size)
    if unit < 1:
        raise ValueError("unit_size must be a positive integer")
    minimum_search = int(minimum_search_rows)
    if minimum_search < 1:
        raise ValueError("minimum_search_rows must be a positive integer")
    minimum_audit = int(minimum_audit_rows)
    if minimum_audit < 1:
        raise ValueError("minimum_audit_rows must be a positive integer")
    requested_rows = int(audit_rows)
    if requested_rows < 0:
        raise ValueError("audit_rows must be nonnegative")
    source = Path(source_path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    source_hash = _sha256_file(source)
    frame = pd.read_csv(source, float_precision="round_trip")
    if _sha256_file(source) != source_hash:
        raise RuntimeError("source CSV changed while the audit firewall was being established")
    target = _validate_ordinary_sr_columns(
        frame, target=target_column, path=source
    )
    n_source = int(len(frame))
    if n_source < minimum_search:
        raise ValueError(
            f"source file has {n_source} rows; at least {minimum_search} search rows are required"
        )

    root = Path(results_dir).expanduser().resolve() / ".stat_selection" / "views"
    root.mkdir(parents=True, exist_ok=True)
    suffix = source.suffix or ".csv"
    digest = source_hash[:12]

    if external_audit_path is not None:
        audit_source = Path(external_audit_path).expanduser().resolve()
        if not audit_source.is_file():
            raise FileNotFoundError(audit_source)
        if audit_source == source:
            raise ValueError("external audit CSV must be distinct from the search CSV")
        audit_hash = _sha256_file(audit_source)
        if audit_hash == source_hash:
            raise ValueError(
                "external audit CSV is byte-identical to the search CSV; an untouched "
                "independent audit sample is required"
            )
        audit_frame = pd.read_csv(audit_source, float_precision="round_trip")
        if _sha256_file(audit_source) != audit_hash:
            raise RuntimeError(
                "external audit CSV changed while the audit firewall was being established"
            )
        audit_target = _validate_ordinary_sr_columns(
            audit_frame,
            target=target,
            path=audit_source,
        )
        if audit_target != target:
            raise ValueError("external audit CSV target does not match the search CSV target")
        if list(audit_frame.columns) != list(frame.columns):
            raise ValueError(
                "external audit CSV columns must exactly match the search CSV columns"
            )
        n_audit = int(len(audit_frame))
        required_audit = max(minimum_audit, 2 * unit)
        if n_audit < required_audit:
            raise ValueError(
                f"external audit file has {n_audit} rows; at least {required_audit} "
                "are required for two declared units"
            )
        if n_audit % unit != 0:
            raise ValueError(
                f"external audit row count {n_audit} is not divisible by unit_size={unit}; "
                "partial audit units are not permitted"
            )
        del audit_frame
        return SRAuditPlan(
            source_path=str(source),
            search_path=str(source),
            audit_path=str(audit_source),
            source_sha256=source_hash,
            search_sha256=source_hash,
            audit_sha256=audit_hash,
            source_rows=n_source,
            search_rows=n_source,
            audit_rows=n_audit,
            audit_kind="external_untouched",
            external_audit_source=str(audit_source),
            target_column=target,
            unit_size=unit,
        )

    if requested_rows == 0:
        fraction = float(audit_fraction)
        if not math.isfinite(fraction) or not 0.0 < fraction < 1.0:
            raise ValueError("audit_fraction must lie strictly between zero and one")
        requested_rows = int(math.ceil(fraction * n_source))
    n_audit = max(minimum_audit, 2 * unit, requested_rows)
    n_audit = int(math.ceil(n_audit / unit) * unit)
    n_search = n_source - n_audit
    if n_search < minimum_search:
        raise ValueError(
            "audit reservation leaves too few search rows: "
            f"source={n_source}, search={n_search}, audit={n_audit}, "
            f"required_search={minimum_search}"
        )

    search_name = f"{source.stem}.stat-search-n{n_search}.{digest}{suffix}"
    audit_name = f"{source.stem}.stat-audit-n{n_search}-{n_source}.{digest}{suffix}"
    search_view = root / search_name
    audit_view = root / audit_name
    # Byte-exact slices: the search must see the original numbers, not a
    # re-serialisation of them.  See _write_row_slice_verbatim.
    _write_row_slice_verbatim(source, search_view, 0, n_search)
    _write_row_slice_verbatim(source, audit_view, n_search, n_source)
    del frame

    return SRAuditPlan(
        source_path=str(source),
        search_path=str(search_view),
        audit_path=str(audit_view),
        source_sha256=source_hash,
        search_sha256=_sha256_file(search_view),
        audit_sha256=_sha256_file(audit_view),
        source_rows=n_source,
        search_rows=n_search,
        audit_rows=n_audit,
        audit_kind="contiguous_tail_untouched",
        audit_start_row=n_search,
        target_column=target,
        unit_size=unit,
    )


def build_sr_candidate_archive(
    *,
    stageB_data: Optional[dict[str, Any]],
    final_polish_summary: Optional[dict[str, Any]],
    max_candidates: int = 1024,
    grammar_version: str = "nestynet-sr-ordinary-v1",
    archive_label: str = "ordinary_sr_global_candidates",
    split_plan: Optional[SRAuditPlan] = None,
) -> SRArchiveBuild:
    """Canonicalise the global ordinary-SR proposal union and freeze it.

    Candidate collection intentionally reuses the CoE collector because that is
    the existing portability boundary for Stage B, Stage C, branch artifacts,
    the proposal reservoir, and final-polish expressions.  Search scores are
    retained only as provenance; the archive complexity is recomputed from the
    symbolic expression and never imports a search scalarisation.
    """

    cap = int(max_candidates)
    if cap < 1:
        raise ValueError("max_candidates must be a positive integer")

    if split_plan is not None:
        observed_search_hash = _sha256_file(split_plan.search_path)
        if observed_search_hash != split_plan.search_sha256:
            raise RuntimeError(
                "search CSV changed after the audit firewall was established"
            )

    from nestynet_sr.equation_polisher import infer_variable_names, parse_sympy_expr
    from nestynet_sr.sr_core.coefficient_metadata import (
        CoefficientMetadataError,
        coefficient_symbol_values_for_expression,
    )
    from nestynet_sr.sr_search.coe_committee import collect_final_candidates

    try:
        import sympy as sp
    except Exception as exc:  # pragma: no cover - project requires SymPy here
        raise RuntimeError("SymPy is required for statistical SR selection") from exc

    upstream_drops: list[dict[str, Any]] = []
    artifacts = collect_final_candidates(
        stageB_data=stageB_data,
        final_polish_summary=final_polish_summary,
        max_candidates=None,
        include_reservoir=True,
        deduplicate=False,
        dropped_log=upstream_drops,
    )
    artifacts = _with_snapped_variants(artifacts)

    excluded: list[dict[str, Any]] = []
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for artifact in artifacts:
        expr_text = str(getattr(artifact, "expr", "") or "").strip()
        if not expr_text:
            continue
        metadata = _json_safe(dict(getattr(artifact, "metadata", {}) or {}))
        unit_certificate = metadata.get("unit_admissibility")
        if isinstance(unit_certificate, dict):
            if unit_certificate.get("checked") is True and unit_certificate.get("valid") is not True:
                excluded.append(
                    {
                        "expression": expr_text,
                        "source": getattr(artifact, "source", "unknown"),
                        "reason": "unit_invalid",
                        "unit_admissibility": unit_certificate,
                    }
                )
                continue
        variable_names = infer_variable_names(expr_text)
        coefficient_metadata = metadata.get("coefficient_metadata")
        try:
            parsed = parse_sympy_expr(expr_text, variable_names)
            symbol_values = coefficient_symbol_values_for_expression(
                coefficient_metadata,
                parsed,
                variable_names=variable_names,
            )
            records_by_symbol = _coefficient_records_by_symbol(
                coefficient_metadata
            )
            substitutions = {}
            fitted_named_parameters = 0
            fixed_contract: list[dict[str, Any]] = []
            for symbol in parsed.free_symbols:
                symbol_name = str(symbol)
                if symbol_name not in symbol_values:
                    continue
                record = records_by_symbol.get(symbol_name, {})
                if str(record.get("kind") or "") == "fixed_const":
                    fixed_contract.append(
                        {
                            "identity": record.get("identity"),
                            "symbol": symbol_name,
                            "value": float(symbol_values[symbol_name]),
                            "dimension": _json_safe(record.get("dimension")),
                            "dimension_status": record.get("dimension_status"),
                            "value_source": record.get("value_source"),
                        }
                    )
                    continue
                substitutions[symbol] = sp.Float(
                    repr(float(symbol_values[symbol_name])), 17
                )
                if bool(record.get("trainable", False)):
                    fitted_named_parameters += 1
            canonical_expr = parsed.xreplace(substitutions)
            canonical_structure = sp.srepr(canonical_expr)
            if fixed_contract:
                canonical_structure += "\nfixed_coefficient_contract=" + json.dumps(
                    sorted(fixed_contract, key=lambda item: str(item.get("identity"))),
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=True,
                )
            display_expr = sp.sstr(parsed)
            n_nodes = int(sum(1 for _ in sp.preorder_traversal(parsed)))
            depth = int(_sympy_depth(parsed))
            constant_code = _constant_code_cost(parsed)
        except CoefficientMetadataError as exc:
            excluded.append(
                {
                    "expression": expr_text,
                    "source": getattr(artifact, "source", "unknown"),
                    "reason": "coefficient_metadata_error",
                    "error_code": exc.code,
                    "error": exc.reason,
                }
            )
            continue
        except Exception as exc:
            excluded.append(
                {
                    "expression": expr_text,
                    "source": getattr(artifact, "source", "unknown"),
                    "reason": "parse_error",
                    "error": str(exc),
                }
            )
            continue
        grouped[canonical_structure].append(
            {
                "display_expr": display_expr,
                "coefficient_metadata": _json_safe(coefficient_metadata),
                "symbol_values": symbol_values,
                "unit_admissibility": unit_certificate,
                "n_nodes": n_nodes,
                "constant_code": constant_code,
                "depth": depth,
                "n_free_params": max(
                    fitted_named_parameters,
                    max(0, int(getattr(artifact, "n_free_params", 0) or 0)),
                ),
                "provenance": {
                    "source": str(getattr(artifact, "source", "unknown") or "unknown"),
                    "label": str(getattr(artifact, "label", "") or ""),
                    "candidate_artifact_id": str(getattr(artifact, "candidate_id", "") or ""),
                    "search_complexity": _json_safe(getattr(artifact, "complexity", None)),
                    "metadata": metadata,
                },
            }
        )

    if not grouped:
        drop_counts: dict[str, int] = {}
        for row in upstream_drops:
            reason = str(row.get("reason", "unknown"))
            drop_counts[reason] = drop_counts.get(reason, 0) + 1
        drop_samples = [
            {k: row.get(k) for k in ("reason", "source", "expr_preview")}
            for row in upstream_drops[:4]
        ]
        detail = excluded[:8]
        raise NoPortableAnalyticCandidatesError(
            "no portable analytic SR candidates were available; "
            f"upstream_drops={drop_counts!r} samples={drop_samples!r}; "
            f"archive_exclusions={detail!r}"
        )

    canonical_records: list[tuple[str, dict[str, Any]]] = []
    for canonical_structure, rows in grouped.items():
        rows_sorted = sorted(
            rows,
            key=lambda row: (
                int(row["n_free_params"]),
                int(row["n_nodes"]),
                int(row["depth"]),
                str(row["display_expr"]),
            ),
        )
        representative = dict(rows_sorted[0])
        representative["n_free_params"] = min(int(row["n_free_params"]) for row in rows)
        representative["provenances"] = _unique_json_records(
            row["provenance"] for row in rows
        )
        canonical_records.append((canonical_structure, representative))

    canonical_records.sort(key=lambda item: _archive_priority(item[1], item[0]))
    cap_applied = len(canonical_records) > cap
    selected_records = canonical_records[:cap]
    cap_policy = (
        "predeclared deterministic proposal-retention cap: certified/final sources, "
        "then free parameters, AST nodes, depth, and canonical hash"
    )
    archive_metadata = {
        "adapter": "ordinary_sr",
        "grammar_version": grammar_version,
        "discovered_artifacts": len(artifacts),
        "canonical_candidates_before_cap": len(canonical_records),
        "max_candidates": cap,
        "cap_applied": cap_applied,
        "cap_policy": cap_policy,
    }
    if split_plan is not None:
        archive_metadata["split_contract_fingerprint"] = split_plan.contract_fingerprint
        archive_metadata["search_sha256"] = split_plan.search_sha256
    archive = CandidateArchive(archive_label=archive_label, metadata=archive_metadata)
    for canonical_structure, row in selected_records:
        complexity = ComplexityVector.from_mapping(
            {
                "ast_nodes": float(row["n_nodes"]),
                "free_parameters": float(row["n_free_params"]),
                "tree_depth": float(row["depth"]),
                "constant_code": float(row["constant_code"]),
            }
        )
        stable_metadata = {
            "expression": row["display_expr"],
            "coefficient_metadata": row["coefficient_metadata"],
            "symbol_values": row["symbol_values"],
            "unit_admissibility": row["unit_admissibility"],
        }
        archive.add_structure(
            canonical_structure,
            complexity,
            grammar_version=grammar_version,
            refit_recipe={
                "kind": "fixed_search_coefficients",
                "audit_refit": False,
                "selection_before_all_data_refit": True,
            },
            metadata=stable_metadata,
            provenance=row["provenances"],
        )
    archive.freeze()
    return SRArchiveBuild(
        archive=archive,
        discovered_count=len(artifacts),
        canonical_count=len(canonical_records),
        excluded=tuple(excluded),
        cap_applied=cap_applied,
        cap_policy=cap_policy,
        upstream_drops=tuple(upstream_drops),
    )


def evaluate_sr_archive(
    archive: CandidateArchive,
    *,
    audit_path: str | Path,
    split_plan: Optional[SRAuditPlan] = None,
    loss_scale: float,
    loss_scale_name: str = "predeclared_search_scale",
    unit_size: int = 1,
    failure_loss: float = 1.0e6,
    target_column: str = "y",
    x_sigma: Any = None,
    x_cov_npz: str | Path | None = None,
    x_cov_sha256_expected: str | None = None,
    x_error_loss: str = "marginal_gaussian_nll",
    x_gradient_step: float = 1.0e-5,
) -> SRAuditEvaluation:
    """Evaluate a frozen archive on one untouched CSV with a common domain.

    The loss is bounded standardized squared error,
    ``min(((prediction-y)/scale)**2, failure_loss)``.  If any row in an
    independent unit is undefined, the entire candidate/unit cell receives the
    same predeclared failure loss.  Thus domain failure cannot improve a model.
    """

    if not archive.frozen:
        raise RuntimeError("freeze the candidate archive before opening audit data")
    pd = _require_pandas()
    from nestynet_sr.equation_polisher import _eval_expr_array, parse_sympy_expr

    path = Path(audit_path).expanduser().resolve()
    if split_plan is not None:
        if str(path) != str(Path(split_plan.audit_path).resolve()):
            raise ValueError("audit path does not match the predeclared split plan")
        observed_hash = _sha256_file(path)
        if observed_hash != split_plan.audit_sha256:
            raise RuntimeError("audit CSV changed after the split plan was established")
    frame = pd.read_csv(path, float_precision="round_trip")
    target = str(target_column)
    _validate_ordinary_sr_columns(frame, target=target, path=path)
    variable_names = [column for column in frame.columns if column != target]
    variable_names.sort(
        key=lambda name: (0, int(name[1:]))
        if str(name).startswith("x") and str(name)[1:].isdigit()
        else (1, str(name))
    )
    X = frame[variable_names].to_numpy(dtype=np.float64)
    y = frame[target].to_numpy(dtype=np.float64).reshape(-1)
    if X.shape[0] != y.size or y.size < 2:
        raise ValueError("audit data must contain at least two aligned rows")
    if not np.all(np.isfinite(y)) or not np.all(np.isfinite(X)):
        raise ValueError("audit inputs and targets must be finite")

    x_cov = None
    x_error_metadata = None
    if x_cov_npz is not None and x_sigma is not None:
        raise ValueError("use either x_sigma or x_cov_npz, not both")
    if x_cov_npz is not None:
        cov_path = Path(x_cov_npz).expanduser().resolve()
        with np.load(cov_path, allow_pickle=False) as bundle:
            if "x_cov" not in bundle.files:
                raise ValueError("x covariance NPZ must contain an x_cov array")
            x_cov = np.asarray(bundle["x_cov"], dtype=np.float64)
        if x_cov.ndim == 2:
            if x_cov.shape != (X.shape[1], X.shape[1]):
                raise ValueError("shared x_cov shape does not match audit input dimension")
            x_cov = np.broadcast_to(x_cov, (X.shape[0],) + x_cov.shape).copy()
        elif x_cov.shape != (X.shape[0], X.shape[1], X.shape[1]):
            raise ValueError("per-row x_cov must have shape (audit_rows,Nx,Nx)")
        observed_cov_hash = _sha256_file(cov_path)
        if x_cov_sha256_expected is not None and observed_cov_hash != str(x_cov_sha256_expected):
            raise RuntimeError("x covariance NPZ changed after the statistical contract was established")
        x_error_metadata = {"source": "npz", "path": str(cov_path), "sha256": observed_cov_hash}
    elif x_sigma is not None:
        if isinstance(x_sigma, str):
            values = [float(v.strip()) for v in x_sigma.split(",") if v.strip()]
            sigma = np.asarray(values, dtype=np.float64)
        else:
            sigma = np.asarray(x_sigma, dtype=np.float64).reshape(-1)
        if sigma.size == 1:
            sigma = np.repeat(sigma, X.shape[1])
        if sigma.size != X.shape[1]:
            raise ValueError("x_sigma must be scalar or contain one value per audit x column")
        if not np.all(np.isfinite(sigma)) or np.any(sigma < 0.0):
            raise ValueError("x_sigma values must be finite and nonnegative")
        x_cov = np.broadcast_to(np.diag(sigma * sigma), (X.shape[0], X.shape[1], X.shape[1])).copy()
        x_error_metadata = {"source": "declared_diagonal_sigma", "sigma": sigma.tolist()}
    x_error_mode = str(x_error_loss).strip().lower()
    if x_error_mode not in {"profile_chi2", "marginal_gaussian_nll"}:
        raise ValueError("x_error_loss must be profile_chi2 or marginal_gaussian_nll")
    gradient_step = float(x_gradient_step)
    if not math.isfinite(gradient_step) or gradient_step <= 0.0:
        raise ValueError("x_gradient_step must be positive and finite")

    scale = float(loss_scale)
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError(
            "loss_scale must be a positive finite value fixed from search data "
            "before the audit is opened"
        )
    scale_name = str(loss_scale_name).strip()
    if not scale_name:
        raise ValueError("loss_scale_name must be non-empty")
    penalty = float(failure_loss)
    if not math.isfinite(penalty) or penalty <= 0.0:
        raise ValueError("failure_loss must be positive and finite")
    block = int(unit_size)
    if block < 1:
        raise ValueError("unit_size must be a positive integer")
    if split_plan is not None:
        if block != int(split_plan.unit_size):
            raise ValueError(
                "unit_size does not match the unit schema sealed in the split plan: "
                f"requested={block}, sealed={split_plan.unit_size}"
            )
        if y.size != int(split_plan.audit_rows):
            raise RuntimeError(
                "audit row count does not match the split plan despite a matching file hash"
            )
    if y.size % block != 0:
        raise ValueError(
            f"audit row count {y.size} is not divisible by unit_size={block}; "
            "partial audit units are not permitted"
        )
    unit_slices = [slice(start, start + block) for start in range(0, y.size, block)]
    if len(unit_slices) < 2:
        raise ValueError("unit_size leaves fewer than two independent audit units")

    candidate_ids = archive.candidate_ids
    losses = np.full((len(unit_slices), len(candidate_ids)), penalty, dtype=np.float64)
    failures = np.ones_like(losses, dtype=np.bool_)
    diagnostics: dict[str, dict[str, Any]] = {}
    predictions: dict[str, Any] = {}
    # Constant-model baseline: the shortest honest description that ignores x.
    with np.errstate(over="ignore", invalid="ignore"):
        null_total_loss = float(np.sum(((y - float(np.mean(y))) / scale) ** 2))
    if not math.isfinite(null_total_loss):
        null_total_loss = 0.0
    row_unit_index = np.zeros(int(y.size), dtype=np.int64)
    for _unit_index, _unit_slice in enumerate(unit_slices):
        row_unit_index[_unit_slice] = _unit_index
    for candidate_index, candidate_id in enumerate(candidate_ids):
        candidate = archive[candidate_id]
        expr_text = str(candidate.metadata.get("expression") or "")
        symbol_values = dict(candidate.metadata.get("symbol_values") or {})
        candidate_diag: dict[str, Any] = {
            "parse_error": None,
            "failed_units": 0,
            "failed_rows": 0,
        }
        try:
            parsed = parse_sympy_expr(expr_text, variable_names)
            pred = np.asarray(
                _eval_expr_array(parsed, X, variable_names, symbol_values=symbol_values),
                dtype=np.float64,
            ).reshape(-1)
            if pred.shape != y.shape:
                raise ValueError(f"prediction shape {pred.shape!r} != target shape {y.shape!r}")
        except Exception as exc:
            candidate_diag["parse_error"] = str(exc)
            candidate_diag["failed_units"] = len(unit_slices)
            candidate_diag["failed_rows"] = int(y.size)
            diagnostics[candidate_id] = candidate_diag
            continue

        invalid_rows = ~np.isfinite(pred)
        effective_variance = np.full(y.shape, scale * scale, dtype=np.float64)
        if x_cov is not None and not np.any(invalid_rows):
            gradients = np.empty((X.shape[0], X.shape[1]), dtype=np.float64)
            for axis in range(X.shape[1]):
                h = gradient_step * np.maximum(1.0, np.abs(X[:, axis]))
                variance_scale = np.sqrt(np.maximum(0.0, x_cov[:, axis, axis]))
                h = np.maximum(h, gradient_step * np.maximum(1.0, variance_scale))
                xp = X.copy(); xm = X.copy()
                xp[:, axis] += h; xm[:, axis] -= h
                try:
                    fp = np.asarray(_eval_expr_array(parsed, xp, variable_names, symbol_values=symbol_values), dtype=np.float64).reshape(-1)
                    fm = np.asarray(_eval_expr_array(parsed, xm, variable_names, symbol_values=symbol_values), dtype=np.float64).reshape(-1)
                    gradients[:, axis] = (fp - fm) / (2.0 * h)
                except Exception:
                    gradients[:, axis] = np.nan
            invalid_rows |= ~np.all(np.isfinite(gradients), axis=1)
            with np.errstate(over="ignore", invalid="ignore"):
                q = np.einsum("bi,bij,bj->b", gradients, x_cov, gradients)
            invalid_rows |= ~np.isfinite(q) | (q < -1.0e-12 * scale * scale)
            effective_variance = scale * scale + np.maximum(q, 0.0)
            candidate_diag["max_x_variance_inflation"] = float(np.nanmax(effective_variance / (scale * scale)))
            candidate_diag["median_x_variance_inflation"] = float(np.nanmedian(effective_variance / (scale * scale)))
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            row_losses = (pred - y) ** 2 / effective_variance
            if x_cov is not None and x_error_mode == "marginal_gaussian_nll":
                row_losses = row_losses + np.log(effective_variance / (scale * scale))
        invalid_rows |= ~np.isfinite(row_losses)
        row_losses = np.minimum(np.where(invalid_rows, penalty, row_losses), penalty)
        for unit_index, unit_slice in enumerate(unit_slices):
            unit_invalid = bool(np.any(invalid_rows[unit_slice]))
            if unit_invalid:
                losses[unit_index, candidate_index] = penalty
                failures[unit_index, candidate_index] = True
            else:
                losses[unit_index, candidate_index] = float(np.mean(row_losses[unit_slice]))
                failures[unit_index, candidate_index] = False
        candidate_diag["failed_units"] = int(np.count_nonzero(failures[:, candidate_index]))
        candidate_diag["failed_rows"] = int(np.count_nonzero(invalid_rows))
        candidate_diag["finite_fraction"] = float(np.mean(~invalid_rows))
        diagnostics[candidate_id] = candidate_diag
        # Retained so functional equivalence can be judged on direct prediction
        # discrepancy.  Equal risk is not equal function, so the losses above
        # cannot answer whether two candidates denote the same law.
        predictions[candidate_id] = np.where(invalid_rows, np.nan, pred)

    unit_ids = tuple(
        f"rows_{unit_slice.start}_{unit_slice.stop}" for unit_slice in unit_slices
    )
    unit_metadata = tuple(
        {
            "row_start": int(unit_slice.start),
            "row_stop": int(unit_slice.stop),
            "n_rows": int(unit_slice.stop - unit_slice.start),
        }
        for unit_slice in unit_slices
    )
    assumptions = [
        "Declared audit units are independent and identically distributed for the target risk.",
        "The untouched audit data were not used to generate, prune, polish, or rank candidates.",
        "Search-fitted continuous coefficients are held fixed during this audit.",
    ]
    if x_cov is not None:
        assumptions.extend([
            "Input errors are locally Gaussian with the declared covariance.",
            "The symbolic expression is locally linear over the support of each input-error covariance.",
            "Schur elimination profiles each latent input displacement independently by audit row.",
        ])
    if split_plan is not None and split_plan.audit_kind == "contiguous_tail_untouched":
        assumptions.append(
            "The reserved contiguous tail is representative of the deployment distribution; "
            "distributional drift across row order would invalidate that interpretation."
        )
    elif split_plan is not None and split_plan.audit_kind == "external_untouched":
        assumptions.append(
            "The external audit sample is representative of the deployment distribution and "
            "was collected independently of candidate search."
        )
    design = AuditDesign(
        loss_name=(
            "bounded_schur_profile_xy_error" if x_cov is not None and x_error_mode == "profile_chi2"
            else "bounded_marginal_gaussian_xy_nll" if x_cov is not None
            else "bounded_standardized_squared_error"
        ),
        unit_kind="iid_row" if block == 1 else f"contiguous_block_of_{block}_rows",
        fit_protocol="fixed coefficients from frozen search archive; no audit refit",
        evaluation_domain={
            "audit_path": str(path),
            "audit_sha256": _sha256_file(path),
            "target_column": target,
            "variable_names": variable_names,
            "row_count": int(y.size),
            "loss_scale": scale,
            "loss_scale_name": scale_name,
            "failure_loss": penalty,
            "split_contract_fingerprint": (
                split_plan.contract_fingerprint if split_plan is not None else None
            ),
            "x_error_model": x_error_metadata,
            "x_error_loss": x_error_mode if x_cov is not None else None,
            "x_gradient_step": gradient_step if x_cov is not None else None,
        },
        sampling_assumptions=tuple(assumptions),
    )
    audit = LossAudit.from_matrix(
        candidate_ids=candidate_ids,
        unit_ids=unit_ids,
        design=design,
        losses=losses,
        failure_mask=failures,
        nonfinite="penalize",
        failure_loss=penalty,
        archive=archive,
        metadata={
            "adapter": "ordinary_sr",
            "bounded_loss": True,
            "common_domain": True,
            "joint_xy_errors": bool(x_cov is not None),
            "schur_profiled_latent_inputs": bool(x_cov is not None),
        },
        unit_metadata=unit_metadata,
    )
    return SRAuditEvaluation(
        audit=audit,
        variable_names=tuple(variable_names),
        scale=scale,
        scale_name=scale_name,
        unit_size=block,
        eligible_candidate_ids=tuple(
            candidate_id
            for index, candidate_id in enumerate(candidate_ids)
            if not bool(np.any(failures[:, index]))
        ),
        candidate_failures=diagnostics,
        predictions=predictions,
        row_unit_index=row_unit_index,
        null_total_standardized_loss=float(null_total_loss),
        n_audit_rows=int(y.size),
    )





def _with_snapped_variants(artifacts):
    """Add constant-snapped variants as first-class candidates, before freezing.

    Snapping an awkward float to an exact constant is a *hypothesis*, not a
    cosmetic rewrite: ``0.3989422804014326`` might be ``1/sqrt(2*pi)``, or it
    might genuinely be that number.  Asserting the snap during rendering makes
    the claim untestable and, if wrong, silently misreports the law.

    Adding the snapped form as its own candidate makes the data decide.  Both
    forms are audited on the same untouched units.  If the snap is right the
    two risks are indistinguishable and the snapped form wins on
    ``constant_code``, dominating strictly.  If the snap is wrong its risk is
    measurably worse and the front keeps the float.

    This runs **before** the archive is frozen, which is what makes it
    legitimate.  Generating these after the audit opened would enlarge the
    comparison family on the strength of data the firewall exists to withhold,
    which is the leak pb018 declined when it refused a Buckingham retry.
    """
    from dataclasses import replace

    from nestynet_sr.equation_polisher import infer_variable_names, parse_sympy_expr
    from nestynet_sr.sr_search.polish_utils import (
        numeric_constant_snap_candidates,
        symbolic_constant_snap_targets,
    )

    out = list(artifacts)
    seen = {str(getattr(a, "expr", "") or "").strip() for a in out}
    targets = symbolic_constant_snap_targets()

    for artifact in list(artifacts):
        expr_text = str(getattr(artifact, "expr", "") or "").strip()
        if not expr_text:
            continue
        try:
            variable_names = infer_variable_names(expr_text)
            parsed = parse_sympy_expr(expr_text, variable_names)
            variants = numeric_constant_snap_candidates(
                parsed, snap_targets=targets, snap_rel_tol=5.0e-4
            )
        except Exception:
            continue
        for label, candidate in variants:
            try:
                text = str(candidate)
            except Exception:
                continue
            if not text or text in seen:
                continue
            seen.add(text)
            metadata = dict(getattr(artifact, "metadata", {}) or {})
            metadata["snapped_from"] = expr_text
            metadata["snap_rule"] = str(label)
            out.append(
                replace(
                    artifact,
                    candidate_id=f"{getattr(artifact, 'candidate_id', 'c')}~snap",
                    expr=text,
                    source=f"{getattr(artifact, 'source', 'unknown')}+snap",
                    metadata=metadata,
                )
            )
    return out


def _constant_code_cost(parsed) -> float:
    """Description length of an expression's numeric constants.

    Node count alone cannot separate ``1*x0`` from ``1.000001*x0``: both have
    the same structure, the same parameter count and the same depth, so the
    dominance order is blind to the difference.  Worse, ``pi/2`` is *more*
    nodes than ``1.5708`` and would score as the more complex of the two, which
    inverts the ordering a physicist wants.

    This charges for the constants themselves, so an exact rational or a named
    constant costs almost nothing while an arbitrary float costs roughly
    ``1.6 + 0.42`` per digit.  Long floats are surcharged again, since a
    sixteen-digit coefficient is a fitted number rather than a law.

    Reuses the cost function the Stage-B display scorer already applies, so the
    archive and the polisher agree about what an awkward constant is.
    """
    try:
        from nestynet_sr.sr_search.polish_utils import (
            constant_code_cost,
            final_polish_snap_targets,
        )

        cost, n_long = constant_code_cost(
            parsed,
            snap_targets=final_polish_snap_targets(),
            snap_rel_tol=1.0e-4,
        )
        return float(cost) + 4.0 * float(n_long)
    except Exception:
        # Never let complexity scoring break archive construction; a zero cost
        # simply restores the previous structure-only behaviour for that row.
        return 0.0





def _identification_complexity_key(
    candidate_id: str, complexity_by_id: Mapping[str, Any]
) -> tuple[Any, ...]:
    """Pre-audit total order used by the structural-identification policy."""

    complexity = complexity_by_id[str(candidate_id)].as_dict()
    return (
        float(complexity.get("free_parameters", math.inf)),
        float(complexity.get("constant_code", math.inf)),
        float(complexity.get("ast_nodes", math.inf)),
        float(complexity.get("tree_depth", math.inf)),
        str(candidate_id),
    )


def _identification_selection(
    audit: LossAudit,
    complexity_by_id: Mapping[str, Any],
    *,
    eligible_candidate_ids: Sequence[str],
    alpha: float,
    delta: float,
    n_resamples: int,
    seed: int,
    multiplier: str,
    numerical_tie_multiplier: float = 128.0,
) -> IdentificationSelectionResult:
    """Walk simple-to-complex; complexity replaces only on certified gain.

    Every possible pair in the frozen total order belongs to the declared
    comparison family.  This lets the incumbent change data-dependently while
    retaining one simultaneous family-wise bound.
    """

    ids = tuple(str(item) for item in audit.candidate_ids)
    if set(ids) != set(complexity_by_id):
        raise ValueError("identification complexity mapping does not match audit")
    eligible_set = {str(item) for item in eligible_candidate_ids}
    if not eligible_set or not eligible_set.issubset(ids):
        raise ValueError("identification requires known eligible candidates")
    alpha_f = float(alpha)
    delta_f = float(delta)
    tie_mult = float(numerical_tie_multiplier)
    if not 0.0 < alpha_f < 1.0:
        raise ValueError("alpha must lie strictly between zero and one")
    if not math.isfinite(delta_f) or delta_f < 0.0:
        raise ValueError("delta must be finite and nonnegative")
    if not math.isfinite(tie_mult) or tie_mult <= 0.0:
        raise ValueError("numerical_tie_multiplier must be positive and finite")

    order = tuple(sorted(ids, key=lambda cid: _identification_complexity_key(cid, complexity_by_id)))
    order_records = tuple({
        "candidate_id": candidate_id,
        "complexity": complexity_by_id[candidate_id].as_dict(),
    } for candidate_id in order)
    eligible_order = tuple(cid for cid in order if cid in eligible_set)
    index = {candidate_id: i for i, candidate_id in enumerate(ids)}
    risks = np.asarray(audit.risks, dtype=np.float64)
    centered = audit.losses - risks[None, :]
    correction = 1.0 - float(np.sum(audit.unit_weights * audit.unit_weights))
    if correction <= 0.0:
        raise ValueError("audit weights do not leave more than one effective unit")

    family_size = len(ids) * (len(ids) - 1) // 2
    regime = select_inference_method(n_units=audit.n_units, k_pre=family_size)
    method = str(regime.get("method") or "bonferroni_t")

    def pair_stats(simple_id: str, complex_id: str) -> dict[str, Any]:
        simple_i, complex_i = index[simple_id], index[complex_id]
        difference = float(risks[simple_i] - risks[complex_i])
        pair_centered = centered[:, simple_i] - centered[:, complex_i]
        denominator = float(
            np.sqrt(np.sum((audit.unit_weights * pair_centered) ** 2))
        )
        pair_scale = max(1.0, abs(difference), float(np.max(np.abs(pair_centered))))
        estimable = denominator > 64.0 * np.finfo(np.float64).eps * pair_scale
        raw_differences = audit.losses[:, simple_i] - audit.losses[:, complex_i]
        loss_scale = max(
            float(np.max(np.abs(audit.losses[:, simple_i]))),
            float(np.max(np.abs(audit.losses[:, complex_i]))),
        )
        scaled_eps = tie_mult * np.finfo(np.float64).eps
        tie_tolerance = max(scaled_eps * scaled_eps, scaled_eps * loss_scale)
        max_abs_difference = float(np.max(np.abs(raw_differences)))
        return {
            "difference": difference,
            "denominator": denominator,
            "standard_error": denominator / math.sqrt(correction),
            "estimable": bool(estimable),
            "numerical_tie": bool(max_abs_difference <= tie_tolerance),
            "numerical_tie_tolerance": float(tie_tolerance),
            "max_abs_unit_loss_difference": max_abs_difference,
        }

    if method == "multiplier_max_t" and family_size:
        simpler_indices: list[int] = []
        complex_indices: list[int] = []
        denominators: list[float] = []
        estimable: list[bool] = []
        for simple_pos, simple_id in enumerate(order):
            for complex_id in order[simple_pos + 1 :]:
                stats = pair_stats(simple_id, complex_id)
                simpler_indices.append(index[simple_id])
                complex_indices.append(index[complex_id])
                denominators.append(float(stats["denominator"]))
                estimable.append(bool(stats["estimable"]))
        critical_value = _max_t_critical_value(
            audit=audit,
            challenger_idx=np.asarray(simpler_indices, dtype=np.int64),
            incumbent_idx=np.asarray(complex_indices, dtype=np.int64),
            denominators=np.asarray(denominators, dtype=np.float64),
            estimable=np.asarray(estimable, dtype=np.bool_),
            alpha=alpha_f,
            n_resamples=int(n_resamples),
            seed=int(seed) + 3,
            multiplier=str(multiplier),
            resample_batch_size=256,
            pair_batch_size=4096,
        )
    else:
        critical_value = _bonferroni_t_critical_value(
            alpha=alpha_f,
            n_comparisons=family_size,
            effective_unit_count=audit.effective_unit_count,
        )

    winner = eligible_order[0]
    challenges: list[dict[str, Any]] = []
    for challenger in eligible_order[1:]:
        incumbent = winner
        stats = pair_stats(incumbent, challenger)
        standard_error = float(stats["standard_error"])
        lower_bound = (
            float(stats["difference"]) - critical_value * standard_error
            if stats["estimable"]
            else None
        )
        if stats["numerical_tie"]:
            decision = "keep_simpler"
            evidence = "numerical_tie"
        elif not stats["estimable"]:
            decision = "keep_simpler"
            evidence = "no_certified_improvement_nonestimable"
        elif lower_bound is not None and lower_bound > delta_f:
            decision = "adopt_more_complex"
            evidence = "complexity_earned_by_certified_improvement"
            winner = challenger
        elif lower_bound is not None and lower_bound > 0.0:
            decision = "keep_simpler"
            evidence = "improvement_certified_but_below_margin"
        else:
            decision = "keep_simpler"
            evidence = "no_certified_improvement"
        challenges.append(
            {
                "incumbent_before": incumbent,
                "challenger": challenger,
                "decision": decision,
                "evidential_state": evidence,
                "risk_improvement_from_added_complexity": float(stats["difference"]),
                "standard_error": standard_error,
                "lower_confidence_bound": lower_bound,
                "minimum_detectable_improvement": (
                    delta_f + critical_value * standard_error
                    if stats["estimable"]
                    else None
                ),
                "numerical_tie": bool(stats["numerical_tie"]),
                "numerical_tie_tolerance": float(stats["numerical_tie_tolerance"]),
                "max_abs_unit_loss_difference": float(
                    stats["max_abs_unit_loss_difference"]
                ),
            }
        )

    return IdentificationSelectionResult(
        candidate_ids=ids,
        complexity_order=order,
        complexity_order_records=order_records,
        selected_candidate_id=winner,
        challenges=tuple(challenges),
        alpha=alpha_f,
        delta=delta_f,
        numerical_tie_multiplier=tie_mult,
        critical_value=float(critical_value),
        critical_value_method=method,
        n_resamples=int(n_resamples),
        seed=int(seed) + 3,
        multiplier=str(multiplier),
        comparison_family_size_pre_audit=family_size,
        inference_regime=regime,
        audit_fingerprint=audit.fingerprint,
    )


def _class_level_audit(*, archive, audit, classes, eligible_candidate_ids):
    """Project the audit onto functional classes, one column per class.

    The front must compare *functions*, not spellings.  Running it over raw
    candidates lets a class with seven encodings contribute seven nodes and
    forty-two ordered comparisons, inflating the multiplicity burden with
    distinctions that carry no scientific content.

    Each class is represented by its minimum-description-length member, and it
    takes that member's *whole* complexity vector.  Taking the best value of
    each component across different members would synthesise a representative
    that does not exist and cannot be printed.
    """
    representatives = [c.representative_id for c in classes]
    index_of = {cid: i for i, cid in enumerate(audit.candidate_ids)}
    columns = [index_of[r] for r in representatives]

    class_ids = [c.class_id for c in classes]
    losses = audit.losses[:, columns]
    failures = audit.failure_mask[:, columns]

    # A class-level archive so the certificate's (archive, audit, result)
    # triple stays internally consistent and its claim reads at the level the
    # front actually compared: functions, not spellings.
    class_archive = CandidateArchive(
        archive_label=f"{archive.archive_label}__functional_classes",
        metadata={
            **dict(archive.metadata or {}),
            "quotiented_by": "exact_algebraic_identity",
            "candidate_archive_fingerprint": archive.fingerprint,
            "n_candidates_before_quotient": len(archive.candidate_ids),
        },
    )
    for cls in classes:
        member = archive[cls.representative_id]
        class_archive.add_structure(
            member.canonical_structure,
            member.complexity,
            grammar_version=member.grammar_version,
            refit_recipe=dict(member.refit_recipe or {}),
            metadata={
                **dict(member.metadata or {}),
                "functional_class": cls.to_dict(),
                "representative_candidate_id": cls.representative_id,
                "class_member_count": len(cls.member_ids),
            },
            candidate_id=cls.class_id,
        )
    class_archive.freeze()

    class_audit = LossAudit.from_matrix(
        candidate_ids=class_ids,
        unit_ids=list(audit.unit_ids),
        design=audit.design,
        losses=losses,
        failure_mask=failures,
        nonfinite="penalize",
        failure_loss=audit.failure_loss,
        metadata={
            **dict(audit.metadata or {}),
            "quotiented_by": "exact_algebraic_identity",
            "n_candidates_before_quotient": len(audit.candidate_ids),
            "n_classes": len(class_ids),
        },
        unit_metadata=list(audit.unit_metadata),
        archive=class_archive,
    )
    complexity = {
        c.class_id: archive[c.representative_id].complexity for c in classes
    }
    eligible = set(eligible_candidate_ids or ())
    eligible_classes = tuple(
        c.class_id for c in classes if c.representative_id in eligible
    )
    representative_of = {c.class_id: c.representative_id for c in classes}
    return class_archive, class_audit, complexity, eligible_classes, representative_of


def _exact_functional_classes(*, archive):
    """Quotient the frozen archive by proven algebraic identity, pre-audit.

    Only this layer may define the hypotheses the front compares: the
    calibration profile requires the comparison family to be fixed before the
    audit is seen, so the partition must derive from frozen expression text
    alone.  Audit-supported near-equivalence is itself an inferential
    conclusion and is reported separately as description, never as
    multiplicity accounting.

    The canonical key is the expanded SymPy form of the frozen expression, so
    printer variants (``x0*x1`` vs ``x1*x0``, reordered sums) merge.  Finite
    decimal literals are interpreted as the exact decimal values they spell:
    in particular, ``0.5`` and ``1/2`` are the same number, whereas
    ``0.3333333333333333`` remains distinct from ``1/3``.  An unparseable
    expression yields no key and stays a singleton, which can only enlarge the
    family.
    """
    from .functional_classes import (
        GRAMMAR_VERSION,
        description_length_bits,
        exact_equivalence_classes,
    )

    ids = list(archive.candidate_ids)
    code_bits: dict[str, float] = {}
    canonical_keys: dict[str, Any] = {}
    for candidate_id in ids:
        record = archive[candidate_id]
        expr_text = str((record.metadata or {}).get("expression") or "")
        bits = float("inf")
        key = None
        if expr_text:
            try:
                import sympy as sp

                from nestynet_sr.equation_polisher import (
                    infer_variable_names,
                    parse_sympy_expr,
                )

                parsed = parse_sympy_expr(expr_text, infer_variable_names(expr_text))
                bits = description_length_bits(
                    parsed,
                    n_free_parameters=int(
                        record.complexity.as_dict().get("free_parameters", 0) or 0
                    ),
                )
                # SymPy deliberately keeps Float(0.5) distinct from
                # Rational(1, 2).  That representation distinction must not
                # manufacture two scientific hypotheses: a finite decimal
                # literal denotes the exact base-10 rational it spells.  Using
                # str(atom), rather than nsimplify, avoids turning a nearby
                # decimal approximation into a different symbolic constant.
                exact_decimals = {
                    atom: sp.Rational(str(atom)) for atom in parsed.atoms(sp.Float)
                }
                exact_parsed = parsed.xreplace(exact_decimals)
                key = sp.srepr(sp.expand(exact_parsed))
            except Exception:
                bits, key = float("inf"), None
        code_bits[candidate_id] = bits
        canonical_keys[candidate_id] = key

    classes = exact_equivalence_classes(ids, canonical_keys, code_bits)
    return classes, code_bits, GRAMMAR_VERSION


def _functional_classes(*, archive, evaluation, delta_function: float):
    """Group candidates by audit-supported near-equivalence, for description.

    Membership is decided on direct prediction discrepancy over the audit
    units, never on equality of risk, because two candidates wrong in opposite
    directions can carry identical risk while denoting different laws.

    Because the partition is learned from the same audit that inference uses,
    it is an inferential conclusion, not preprocessing: these classes must not
    define the comparison family, the front, or the multiplicity accounting.
    The inferential quotient is :func:`_exact_functional_classes`; this one is
    reported in the certificate as presentation only.
    """
    from .functional_classes import (
        GRAMMAR_VERSION,
        certified_equivalence_classes,
        description_length_bits,
        prediction_discrepancy,
    )

    ids = list(archive.candidate_ids)
    preds = evaluation.predictions or {}
    row_index = evaluation.row_unit_index
    n_units = int(evaluation.audit.n_units)

    code_bits: dict[str, float] = {}
    for candidate_id in ids:
        record = archive[candidate_id]
        expr_text = str((record.metadata or {}).get("expression") or "")
        bits = float("inf")
        if expr_text:
            try:
                from nestynet_sr.equation_polisher import (
                    infer_variable_names,
                    parse_sympy_expr,
                )

                parsed = parse_sympy_expr(expr_text, infer_variable_names(expr_text))
                bits = description_length_bits(
                    parsed,
                    n_free_parameters=int(
                        record.complexity.as_dict().get("free_parameters", 0) or 0
                    ),
                )
            except Exception:
                bits = float("inf")
        code_bits[candidate_id] = bits

    discrepancy: dict[tuple[str, str], float] = {}
    if row_index is not None:
        for i, a in enumerate(ids):
            for b in ids[i + 1:]:
                pa, pb = preds.get(a), preds.get(b)
                if pa is None or pb is None:
                    discrepancy[(a, b)] = float("inf")
                    continue
                mean, se = prediction_discrepancy(
                    pa, pb, scale=float(evaluation.scale),
                    unit_index=row_index, n_units=n_units,
                )
                # Upper bound, so a class is only formed when the data cannot
                # separate the two even allowing for sampling error.
                discrepancy[(a, b)] = float(mean + 1.645 * se)

    classes = certified_equivalence_classes(
        ids, discrepancy, code_bits, delta_function=float(delta_function)
    )
    return classes, code_bits, GRAMMAR_VERSION


def _admissible_comparison_count(candidate_ids, complexity_by_id) -> int:
    """Count ordered pairs the dominance machinery would consider.

    A pair is admissible when the challenger is no more complex than the
    incumbent, which is the same filter ``confidence_pareto`` applies.
    """
    ids = list(candidate_ids)
    total = 0
    for incumbent in ids:
        for challenger in ids:
            if challenger == incumbent:
                continue
            if complexity_by_id[challenger].no_worse_than(
                complexity_by_id[incumbent], atol=0.0
            ):
                total += 1
    return int(total)


def _inference_regime(*, archive, audit, eligible_candidate_ids) -> dict[str, Any]:
    """Record the calibration lookup and the family sizes it was keyed on.

    The multiplicity burden must be keyed to the **pre-audit** comparison
    family, computed from frozen candidate structure and declared complexity
    alone.  Keying it to the estimable subset would let a candidate shrink the
    burden by failing on the audit, so a pathological candidate could make the
    method appear to have faced a smaller family after the data were seen.
    Non-estimable comparisons stay in the family and are treated
    conservatively; they simply cannot generate a dominance edge.
    """
    complexity_by_id = archive.complexity_by_id()
    all_ids = list(archive.candidate_ids)
    eligible = list(eligible_candidate_ids)

    k_pre = _admissible_comparison_count(all_ids, complexity_by_id)
    k_estimable = _admissible_comparison_count(eligible, complexity_by_id)
    decision = select_inference_method(n_units=int(audit.n_units), k_pre=k_pre)

    return {
        "independent_units": int(audit.n_units),
        "comparison_family_size_pre_audit": int(k_pre),
        "comparison_family_size_estimable": int(k_estimable),
        "comparison_family_size_nonestimable": int(k_pre - k_estimable),
        "calibration_lookup_key": [int(audit.n_units), int(k_pre)],
        "n_candidates_frozen": len(all_ids),
        "n_candidates_eligible": len(eligible),
        **decision,
    }


def certify_sr_archive(
    *,
    archive_build: SRArchiveBuild,
    evaluation: SRAuditEvaluation,
    output_dir: str | Path,
    archive_filename: str = "sr_candidate_archive.json",
    certificate_filename: str = "sr_pareto_certificate.json",
    alpha: float = 0.05,
    delta: float = 0.0,
    n_resamples: int = 4000,
    seed: int = 12345,
    multiplier: str = "normal",
    inclusion_resamples: Optional[int] = None,
    delta_function: Optional[float] = None,
) -> SRSelectionOutcome:
    """Construct the simultaneous front, choose transparently, and persist it."""

    archive = archive_build.archive
    audit = evaluation.audit
    if not evaluation.eligible_candidate_ids:
        raise RuntimeError(
            "no candidate was finite on every declared audit unit; the feasibility "
            "front is empty"
        )
    classes, code_bits, grammar = _exact_functional_classes(archive=archive)
    (
        class_archive,
        class_audit,
        class_complexity,
        eligible_classes,
        representative_of,
    ) = _class_level_audit(
        archive=archive,
        audit=audit,
        classes=classes,
        eligible_candidate_ids=evaluation.eligible_candidate_ids,
    )

    # The regime is decided before the front is built, on the pre-audit family
    # over exact classes, and its method is dispatched rather than merely
    # recorded: outside the calibrated envelope the Bonferroni-t critical
    # value is what actually bounds the edges.
    regime = _inference_regime(
        archive=class_archive,
        audit=class_audit,
        eligible_candidate_ids=eligible_classes,
    )
    inference_method = str(regime.get("method") or "multiplier_max_t")

    result = confidence_pareto(
        class_audit,
        class_complexity,
        alpha=alpha,
        delta=delta,
        n_resamples=n_resamples,
        seed=seed,
        multiplier=multiplier,
        eligible_candidate_ids=eligible_classes,
        method=inference_method,
        bonferroni_comparisons=int(regime["comparison_family_size_pre_audit"]),
    )
    deployment = _deployment_noninferiority_set(
        class_audit,
        alpha=alpha,
        delta=delta,
        n_resamples=n_resamples,
        seed=int(seed) + 2,
        multiplier=multiplier,
    )
    if not deployment.noninferiority_set:
        raise RuntimeError("deployment noninferiority construction returned an empty set")
    identification = _identification_selection(
        class_audit,
        class_complexity,
        eligible_candidate_ids=eligible_classes,
        alpha=alpha,
        delta=delta,
        n_resamples=n_resamples,
        seed=seed,
        multiplier=multiplier,
    )
    selected_class = identification.selected_candidate_id
    selected_id = representative_of[selected_class]
    n_inclusion = int(inclusion_resamples or min(max(250, n_resamples // 2), 2000))
    inclusion = bootstrap_front_inclusion_frequencies(
        class_audit,
        class_complexity,
        n_resamples=n_inclusion,
        seed=int(seed) + 1,
        delta=float(delta),
        eligible_candidate_ids=eligible_classes,
    )
    out_dir = Path(output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    archive_path = archive.write_json(out_dir / archive_filename)
    certificate = build_certificate(class_archive, class_audit, result)
    certificate_path = out_dir / certificate_filename
    certificate_payload = certificate.to_dict()
    certificate_payload["ordinary_sr_deployment"] = deployment.to_dict()
    certificate_payload["ordinary_sr_identification"] = identification.to_dict()
    try:
        certificate_payload["functional_classes"] = {
            "grammar_version": grammar,
            "equivalence_kind": "exact_algebraic",
            "canonicalisation": "sympy_expand_srepr_of_frozen_expression",
            "n_classes": len(classes),
            "n_candidates": len(archive.candidate_ids),
            "classes": [c.to_dict() for c in classes],
            "note": (
                "Classes merge spellings whose frozen expressions canonicalise "
                "to the same algebraic form, so the partition is fixed before "
                "the audit is seen and the comparison family may be counted "
                "over classes. The representative is the minimum-description-"
                "length member, a naming decision and not a claim of better "
                "support."
            ),
        }
    except Exception as exc:
        certificate_payload["functional_classes"] = {"error": str(exc)}

    try:
        from .functional_classes import derive_delta_function

        near_delta = (
            float(delta_function)
            if delta_function is not None
            else derive_delta_function(
                n_rows=int(evaluation.n_audit_rows or evaluation.audit.n_units),
                code_gap_bits=1.0,
            )
        )
        near_classes, _, _ = _functional_classes(
            archive=archive, evaluation=evaluation, delta_function=near_delta,
        )
        certificate_payload["near_equivalence_descriptive"] = {
            "grammar_version": grammar,
            "role": "descriptive_only",
            "delta_function": float(near_delta),
            "delta_function_rule": (
                "caller-declared"
                if delta_function is not None
                else "(code_gap_bits * ln2 / n_rows)^2 with code_gap_bits=1"
            ),
            "discrepancy_rule": "upper_bound_on_per_unit_prediction_discrepancy",
            "linkage": "complete",
            "n_classes": len(near_classes),
            "classes": [c.to_dict() for c in near_classes],
            "note": (
                "Audit-based near-equivalence is an inferential conclusion "
                "drawn from the same audit used for selection. It is reported "
                "for presentation only and plays no role in the comparison "
                "family, the front, or the multiplicity accounting."
            ),
        }
    except Exception as exc:
        certificate_payload["near_equivalence_descriptive"] = {"error": str(exc)}

    try:
        from .functional_classes import compression_certificate

        # Risks are keyed by class. Encode the authoritative identification
        # winner, using its representative candidate only to print the law.
        class_risks = result.risk_by_id()
        best_class = selected_class
        best_id = representative_of[best_class]
        best_expr = str((archive[best_id].metadata or {}).get("expression") or "")
        model_bits = float(code_bits.get(best_id, math.inf))
        if not math.isfinite(model_bits):
            raise RuntimeError(
                f"no finite description length for representative {best_id!r}"
            )
        n_rows = int(evaluation.n_audit_rows or evaluation.audit.n_units)
        total_loss = float(class_risks[best_class]) * float(n_rows)
        certificate_payload["compression"] = compression_certificate(
            model_expression=best_expr,
            model_code_bits=model_bits,
            total_standardized_loss=total_loss,
            null_total_standardized_loss=float(
                evaluation.null_total_standardized_loss
            ),
            n_rows=n_rows,
            sigma_source=str(evaluation.scale_name),
            null_model_code_bits=32.0,
        ).to_dict()
    except Exception as exc:
        certificate_payload["compression"] = {"error": str(exc)}

    # The multiplicity burden is over the comparisons actually made, and the
    # front compares functions.  Counting spellings would charge the procedure
    # for distinctions that carry no scientific content.
    certificate_payload["inference_regime"] = dict(regime)
    certificate_payload["inference_regime"]["quotiented_by"] = (
        "exact_algebraic_identity"
    )
    certificate_payload["inference_regime"]["n_candidates_before_quotient"] = len(
        archive.candidate_ids
    )
    # Proof of dispatch: the method the lookup licensed and the critical value
    # the front actually used, side by side.
    certificate_payload["inference_regime"]["method_executed"] = (
        result.critical_value_method
    )
    certificate_payload["inference_regime"]["critical_value"] = float(
        result.critical_value
    )
    certificate_path.write_text(
        json.dumps(
            certificate_payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return SRSelectionOutcome(
        archive_build=archive_build,
        evaluation=evaluation,
        pareto=result,
        deployment=deployment,
        identification=identification,
        selected_candidate_id=selected_id,
        front_inclusion_frequencies=inclusion,
        functional_classes=[c.to_dict() for c in classes],
        archive_path=str(archive_path),
        certificate_path=str(certificate_path),
    )


def run_sr_statistical_selection(
    *,
    stageB_data: Optional[dict[str, Any]],
    final_polish_summary: Optional[dict[str, Any]],
    split_plan: SRAuditPlan,
    output_dir: str | Path,
    loss_scale: float,
    loss_scale_name: str = "predeclared_search_scale",
    max_candidates: int = 1024,
    unit_size: int = 1,
    failure_loss: float = 1.0e6,
    alpha: float = 0.05,
    delta: float = 0.0,
    n_resamples: int = 4000,
    seed: int = 12345,
    multiplier: str = "normal",
    archive_filename: str = "sr_candidate_archive.json",
    certificate_filename: str = "sr_pareto_certificate.json",
    x_sigma: Any = None,
    x_cov_npz: str | Path | None = None,
    x_cov_sha256_expected: str | None = None,
    x_error_loss: str = "marginal_gaussian_nll",
    x_gradient_step: float = 1.0e-5,
) -> SRSelectionOutcome:
    """End-to-end ordinary-SR archive, untouched audit, and certification."""

    archive_build = build_sr_candidate_archive(
        stageB_data=stageB_data,
        final_polish_summary=final_polish_summary,
        max_candidates=max_candidates,
        split_plan=split_plan,
    )
    evaluation = evaluate_sr_archive(
        archive_build.archive,
        audit_path=split_plan.audit_path,
        split_plan=split_plan,
        loss_scale=loss_scale,
        loss_scale_name=loss_scale_name,
        unit_size=unit_size,
        failure_loss=failure_loss,
        target_column=split_plan.target_column,
        x_sigma=x_sigma,
        x_cov_npz=x_cov_npz,
        x_cov_sha256_expected=x_cov_sha256_expected,
        x_error_loss=x_error_loss,
        x_gradient_step=x_gradient_step,
    )
    return certify_sr_archive(
        archive_build=archive_build,
        evaluation=evaluation,
        output_dir=output_dir,
        archive_filename=archive_filename,
        certificate_filename=certificate_filename,
        alpha=alpha,
        delta=delta,
        n_resamples=n_resamples,
        seed=seed,
        multiplier=multiplier,
    )


def update_report_with_sr_statistical_selection(
    report_path: str | Path,
    outcome: SRSelectionOutcome,
    *,
    split_plan: SRAuditPlan,
) -> dict[str, Any]:
    """Make statistical selection authoritative while preserving legacy choices."""

    path = Path(report_path)
    report = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(report, dict):
        raise ValueError("report root is not a JSON object")
    summary = _json_safe(outcome.summary())
    summary["split_plan"] = split_plan.to_dict()
    summary["split_plan_fingerprint"] = split_plan.fingerprint
    summary["split_contract_fingerprint"] = split_plan.contract_fingerprint
    old_selection = report.get("final_selection")
    if isinstance(old_selection, dict):
        report["legacy_search_selection"] = old_selection
    selected = outcome.archive_build.archive[outcome.selected_candidate_id]
    final_selection = {
        "source": "statistical_selection",
        "mode": "confidence_pareto_with_occam_identification_selection",
        "applied": True,
        "eligible_for_success": True,
        "status": "certified",
        "candidate_id": outcome.selected_candidate_id,
        "candidate_source": "frozen_global_sr_archive",
        "selection_basis": summary["selection_basis"],
        "expr": selected.metadata.get("expression"),
        # Risk is a property of the function, so it is looked up by class.
        "risk": summary.get("selected_risk"),
        "functional_class": summary.get("selected_functional_class"),
        "complexity": selected.complexity.as_dict(),
        "coefficient_metadata": _json_safe(
            selected.metadata.get("coefficient_metadata")
        ),
        "unit_admissibility": _json_safe(selected.metadata.get("unit_admissibility")),
        "certificate_path": outcome.certificate_path,
        "archive_path": outcome.archive_path,
    }
    report["statistical_selection"] = summary
    report["final_selection"] = final_selection
    path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return summary


def format_sr_statistical_selection(summary: Mapping[str, Any]) -> str:
    """Compact human-readable summary for the path and final-human reports."""

    if not isinstance(summary, Mapping) or not summary.get("enabled"):
        return ""
    lines = ["=== Statistical Confidence Pareto Selection ==="]
    lines.append(f"status: {summary.get('status')}")
    lines.append(f"candidates: {summary.get('n_candidates')}")
    lines.append(f"independent_units: {summary.get('n_units')}")
    lines.append(f"alpha: {summary.get('alpha')}")
    lines.append(f"delta: {summary.get('delta')}")
    lines.append(f"selected_candidate_id: {summary.get('selected_candidate_id')}")
    lines.append(f"selected_risk: {summary.get('selected_risk')}")
    lines.append(f"selected_expression: {summary.get('selected_expression')}")
    lines.append(f"point_front: {', '.join(summary.get('point_front') or [])}")
    lines.append(f"confidence_front: {', '.join(summary.get('confidence_front') or [])}")
    lines.append(f"practical_front: {', '.join(summary.get('practical_front') or [])}")
    identification = summary.get("identification_selection") or {}
    lines.append(
        "identification_complexity_order: "
        + ", ".join(identification.get("complexity_order") or [])
    )
    deployment = summary.get("deployment_noninferiority") or {}
    lines.append(
        "deployment_noninferiority_set: "
        + ", ".join(deployment.get("noninferiority_set") or [])
    )
    lines.append(f"certificate: {summary.get('certificate_path')}")
    return "\n".join(lines)


def _deployment_noninferiority_set(
    audit: LossAudit,
    *,
    alpha: float,
    delta: float,
    n_resamples: int,
    seed: int,
    multiplier: str,
    resample_batch_size: int = 256,
) -> DeploymentNoninferiorityResult:
    """Return a scalable all-candidate simultaneous risk noninferiority set.

    For each multiplier draw, the largest ordered pair perturbation is simply
    ``max(z) - min(z)``.  This gives an all-pairs one-sided max-range bound in
    O(B M), rather than materialising O(B M^2) studentised comparisons.  The
    ordinary-SR loss is pre-standardised and bounded, making this conservative
    unstudentised construction a useful deployment firewall.
    """

    alpha_f = float(alpha)
    delta_f = float(delta)
    n_boot = int(n_resamples)
    batch_size = max(1, int(resample_batch_size))
    multiplier_name = str(multiplier).strip().lower()
    if not 0.0 < alpha_f < 1.0:
        raise ValueError("alpha must lie strictly between zero and one")
    if not math.isfinite(delta_f) or delta_f < 0.0:
        raise ValueError("delta must be finite and nonnegative")
    if n_boot < 1:
        raise ValueError("n_resamples must be positive")
    if multiplier_name not in {"normal", "rademacher"}:
        raise ValueError("multiplier must be 'normal' or 'rademacher'")

    candidate_ids = audit.candidate_ids
    risks = np.asarray(audit.risks, dtype=np.float64)
    failure_counts = np.sum(audit.failure_mask, axis=0)
    eligible = np.asarray(failure_counts == 0, dtype=np.bool_)
    if not np.any(eligible):
        raise RuntimeError("no candidate is finite on every declared audit unit")
    eligible_indices = np.flatnonzero(eligible)
    reference_index = min(
        (int(index) for index in eligible_indices),
        key=lambda index: (float(risks[index]), candidate_ids[index]),
    )
    reference_id = candidate_ids[reference_index]

    centered = audit.losses - risks[None, :]
    weighted_centered = audit.unit_weights[:, None] * centered
    correction = 1.0 - float(np.sum(audit.unit_weights * audit.unit_weights))
    if correction <= 0.0:
        raise ValueError("audit weights do not leave more than one effective unit")
    rng = np.random.default_rng(int(seed))
    maxima = np.empty(n_boot, dtype=np.float64)
    written = 0
    while written < n_boot:
        batch = min(batch_size, n_boot - written)
        if multiplier_name == "normal":
            multipliers = rng.standard_normal((batch, audit.n_units))
        else:
            multipliers = rng.integers(
                0, 2, size=(batch, audit.n_units), dtype=np.int8
            )
            multipliers = multipliers.astype(np.float64) * 2.0 - 1.0
        perturbations = multipliers @ weighted_centered[:, eligible]
        maxima[written : written + batch] = (
            np.max(perturbations, axis=1) - np.min(perturbations, axis=1)
        ) / math.sqrt(correction)
        written += batch
    critical_radius = _higher_quantile(maxima, 1.0 - alpha_f)
    critical_radius = float(max(0.0, critical_radius))

    differences = risks - risks[reference_index]
    standard_errors = np.empty(audit.n_candidates, dtype=np.float64)
    estimable = np.zeros(audit.n_candidates, dtype=np.bool_)
    upper_bounds = np.full(audit.n_candidates, np.inf, dtype=np.float64)
    reference_centered = centered[:, reference_index]
    for index in range(audit.n_candidates):
        if index == reference_index:
            standard_errors[index] = 0.0
            estimable[index] = True
            upper_bounds[index] = 0.0
            continue
        pair_centered = centered[:, index] - reference_centered
        denominator = float(
            np.sqrt(np.sum((audit.unit_weights * pair_centered) ** 2))
        )
        pair_scale = max(
            1.0,
            abs(float(differences[index])),
            float(np.max(np.abs(pair_centered))),
        )
        floor = 64.0 * np.finfo(np.float64).eps * pair_scale
        standard_errors[index] = denominator / math.sqrt(correction)
        estimable[index] = denominator > floor
        if eligible[index] and estimable[index]:
            upper_bounds[index] = float(differences[index] + critical_radius)

    included = [reference_id]
    for index, candidate_id in enumerate(candidate_ids):
        if index == reference_index:
            continue
        if eligible[index] and estimable[index] and upper_bounds[index] <= delta_f:
            included.append(candidate_id)
    included.sort(
        key=lambda candidate_id: (
            float(risks[candidate_ids.index(candidate_id)]), candidate_id
        )
    )

    warnings: list[str] = []
    if audit.effective_unit_count < 20.0:
        warnings.append(
            "Fewer than 20 effective independent units were supplied; deployment-set "
            "coverage should be checked by problem-specific simulation."
        )
    if np.any(~eligible):
        warnings.append(
            "Candidates with any common-domain audit failure were retained in the Pareto "
            "certificate but excluded from deployment eligibility."
        )
    if np.any(eligible & ~estimable):
        warnings.append(
            "At least one non-reference comparison had empirically degenerate paired "
            "variance and was excluded from the deployment set conservatively."
        )
    if n_boot < 2000:
        warnings.append(
            "Fewer than 2000 multiplier draws were requested; deployment-set tail "
            "quantile Monte Carlo error may be visible."
        )
    return DeploymentNoninferiorityResult(
        candidate_ids=candidate_ids,
        reference_candidate_id=reference_id,
        noninferiority_set=tuple(included),
        risk_differences=tuple(float(value) for value in differences),
        standard_errors=tuple(float(value) for value in standard_errors),
        upper_confidence_bounds=tuple(float(value) for value in upper_bounds),
        estimable=tuple(bool(value) for value in estimable),
        eligible=tuple(bool(value) for value in eligible),
        alpha=alpha_f,
        delta=delta_f,
        critical_radius=critical_radius,
        n_resamples=n_boot,
        seed=int(seed),
        multiplier=multiplier_name,
        audit_fingerprint=audit.fingerprint,
        warnings=tuple(warnings),
    )


def _higher_quantile(values: np.ndarray, probability: float) -> float:
    try:
        return float(np.quantile(values, probability, method="higher"))
    except TypeError:  # pragma: no cover - NumPy < 1.22
        return float(np.quantile(values, probability, interpolation="higher"))




def _coefficient_records_by_symbol(payload: Any) -> dict[str, Mapping[str, Any]]:
    if not isinstance(payload, Mapping):
        return {}
    records = payload.get("records")
    if not isinstance(records, (list, tuple)):
        return {}
    out: dict[str, Mapping[str, Any]] = {}
    for record in records:
        if not isinstance(record, Mapping):
            continue
        symbol = record.get("symbol")
        if symbol is not None:
            out[str(symbol)] = record
    return out


def _archive_priority(row: Mapping[str, Any], canonical_structure: str) -> tuple[Any, ...]:
    sources = [str(item.get("source", "")) for item in row.get("provenances", [])]
    priority = 3
    if any(source.startswith("stageC") for source in sources):
        priority = 0
    elif any(source.startswith("final_polish:recommended") for source in sources):
        priority = 0
    elif any(source.startswith("final_polish:seed") for source in sources):
        priority = 1
    elif any("stageB_y_branch" in source for source in sources):
        priority = 1
    return (
        priority,
        int(row["n_free_params"]),
        int(row["n_nodes"]),
        int(row["depth"]),
        hashlib.sha256(canonical_structure.encode("utf-8")).hexdigest(),
    )


def _sympy_depth(expr: Any) -> int:
    args = getattr(expr, "args", ())
    if not args:
        return 1
    return 1 + max(_sympy_depth(arg) for arg in args)


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_ordinary_sr_columns(
    frame: Any, *, target: str | None, path: Path
) -> str:
    columns = [str(column) for column in frame.columns]
    target_columns = [column for column in columns if column.startswith("y")]
    if target is None:
        if len(target_columns) != 1:
            raise ValueError(
                "installment 2 ordinary-SR certification requires exactly one "
                f"y-prefixed target column; found {target_columns!r} in {path}"
            )
        resolved_target = target_columns[0]
    else:
        resolved_target = str(target)
        if resolved_target not in columns:
            raise ValueError(f"target column {resolved_target!r} not found in {path}")
        if target_columns != [resolved_target]:
            raise ValueError(
                "installment 2 ordinary-SR certification requires exactly one y-prefixed "
                f"target column {resolved_target!r}; found {target_columns!r} in {path}"
            )
    if len(columns) < 2:
        raise ValueError("ordinary-SR certification requires at least one input column")
    try:
        values = frame.to_numpy(dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"installment 2 ordinary-SR certification requires numeric columns in {path}"
        ) from exc
    if values.ndim != 2 or values.shape[1] != len(columns):
        raise ValueError(f"could not form a rectangular numeric table from {path}")
    if not np.all(np.isfinite(values)):
        raise ValueError(
            f"installment 2 ordinary-SR certification requires finite inputs and targets in {path}"
        )
    return resolved_target



def _write_row_slice_verbatim(source: Path, path: Path, start: int, stop: int) -> None:
    """Write rows [start, stop) of ``source`` by copying its bytes.

    The view must be byte-identical to the source rows, not a re-serialisation.
    Round-tripping through a parser is lossy: pandas' default C float parser is
    fast but not correctly rounded and can be off by one ULP, which measured
    3.6e-15 on ~13 percent of values in the AI Feynman tables.  That is
    negligible against measurement noise and catastrophic on a noiseless
    benchmark, where the ordinary path reaches a Stage-B loss of 1e-30: a
    perturbation of the *inputs* far larger than the residual being minimised
    stops the fit reaching exactness, and the separability structure that
    depends on it is no longer detected.

    Copying bytes also makes the recorded SHA-256 mean what it claims, since it
    then hashes the data the search actually saw rather than a re-encoding of
    it.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        with open(source, "r", encoding="utf-8", newline="") as src, \
                open(temporary, "w", encoding="utf-8", newline="") as dst:
            header = src.readline()
            if not header:
                raise ValueError(f"source CSV is empty: {source}")
            dst.write(header)
            for index, line in enumerate(src):
                if index >= stop:
                    break
                if index >= start:
                    dst.write(line)
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _atomic_write_csv(frame: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        frame.to_csv(temporary, index=False, float_format="%.17g")
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _unique_json_records(records: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    unique: dict[str, dict[str, Any]] = {}
    for record in records:
        safe = dict(_json_safe(record))
        key = json.dumps(safe, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        unique.setdefault(key, safe)
    return [unique[key] for key in sorted(unique)]


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    try:
        scalar = value.item()
    except Exception:
        scalar = value
    if scalar is not value:
        return _json_safe(scalar)
    try:
        return str(value)
    except Exception:
        return repr(value)


def _require_pandas():
    try:
        import pandas as pd
    except Exception as exc:  # pragma: no cover - dependency is required by SR runner
        raise RuntimeError("pandas is required for ordinary-SR statistical selection") from exc
    return pd
