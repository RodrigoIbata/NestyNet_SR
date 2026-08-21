# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata

"""Run-level reporting for the generalized-symmetry (GS) layer."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
import json
import math
import os
import threading

_LOCK = threading.RLock()
_STAGEA_EVENTS: list[dict[str, Any]] = []
_JET_EVENTS: list[dict[str, Any]] = []
_POLICY_EVENTS: list[dict[str, Any]] = []
_DE_EVENTS: list[dict[str, Any]] = []
_UNIT_TORUS_EVENTS: list[dict[str, Any]] = []
_RUN_METADATA: dict[str, Any] = {}


def reset_gs_reporter(metadata: dict[str, Any] | None = None) -> None:
    """Clear in-memory GS events for a fresh run."""

    with _LOCK:
        _STAGEA_EVENTS.clear()
        _JET_EVENTS.clear()
        _POLICY_EVENTS.clear()
        _DE_EVENTS.clear()
        _UNIT_TORUS_EVENTS.clear()
        _RUN_METADATA.clear()
        if metadata:
            _RUN_METADATA.update(_jsonable(metadata))


def configure_gs_reporter(**metadata: Any) -> None:
    """Update run metadata without clearing accumulated events."""

    with _LOCK:
        _RUN_METADATA.update(_jsonable(metadata))


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "detach"):
        try:
            return _jsonable(value.detach().cpu().tolist())
        except Exception:
            return str(value)
    try:
        return _jsonable(asdict(value))
    except Exception:
        return str(value)


def record_stagea_event(
    *,
    cols: Iterable[int] | None,
    diagnostics: list[dict[str, Any]] | None,
    proposals: list[Any] | None = None,
    context: dict[str, Any] | None = None,
) -> None:
    """Record one Stage-A affine-generator probe."""

    diagnostics = list(diagnostics or [])
    if not diagnostics and not proposals:
        return
    prop_summaries: list[dict[str, Any]] = []
    for p in list(proposals or []):
        try:
            pattern, _z_ast, confidence, _extra, meta = p
            prop_summaries.append(
                {
                    "pattern": list(pattern),
                    "confidence": float(confidence),
                    "kind": str((meta or {}).get("kind", "")),
                    "family": str((meta or {}).get("gs_family", "")),
                    "gs_kind": str((meta or {}).get("gs_kind", "")),
                    "axes": list((meta or {}).get("gs_axes", [])),
                    "invariant": (meta or {}).get("gs_invariant", None),
                    "z_human": (meta or {}).get("z_human", None),
                    "residual_rel": (meta or {}).get("gs_residual_rel", None),
                }
            )
        except Exception:
            prop_summaries.append({"repr": repr(p)})
    with _LOCK:
        _STAGEA_EVENTS.append(
            {
                "event_index": len(_STAGEA_EVENTS),
                "cols": list(cols or []),
                "diagnostics": _jsonable(diagnostics),
                "proposals": _jsonable(prop_summaries),
                "context": _jsonable(context or {}),
            }
        )


def record_jet_event(
    *,
    diagnostics: Iterable[dict[str, Any]] | None,
    proposals: Iterable[Any] | None = None,
    context: dict[str, Any] | None = None,
) -> None:
    """Record a jet-level separability probe."""

    diagnostics = list(diagnostics or [])
    if not diagnostics and not proposals:
        return
    prop_summaries: list[dict[str, Any]] = []
    for p in list(proposals or []):
        try:
            op, g1, g2, offset, metric = p
            name = getattr(op, "__name__", str(op))
            if name == "add":
                op_name = "additive"
            elif name == "multiply":
                op_name = "multiplicative"
            else:
                op_name = str(name)
            prop_summaries.append(
                {
                    "op": op_name,
                    "group1": _jsonable(list(g1 or [])),
                    "group2": _jsonable(list(g2 or [])),
                    "offset": _jsonable(offset),
                    "metric": _jsonable(metric),
                }
            )
        except Exception:
            prop_summaries.append({"repr": repr(p)})
    with _LOCK:
        _JET_EVENTS.append(
            {
                "event_index": len(_JET_EVENTS),
                "diagnostics": _jsonable(diagnostics),
                "proposals": _jsonable(prop_summaries),
                "context": _jsonable(context or {}),
            }
        )



def record_policy_event(*, policy: str, action: str, details: dict[str, Any] | None = None) -> None:
    """Record GS policy-side effects, such as baseline proposal suppression."""
    with _LOCK:
        _POLICY_EVENTS.append({
            "event_index": len(_POLICY_EVENTS),
            "policy": str(policy),
            "action": str(action),
            "details": _jsonable(details or {}),
        })



def record_unit_torus_event(
    *,
    event_type: str,
    diagnostics: Iterable[dict[str, Any]] | None = None,
    proposals: Iterable[Any] | None = None,
    decisions: Iterable[Any] | None = None,
    context: dict[str, Any] | None = None,
) -> None:
    """Record unit-torus generators, pi proposals, or dimensional decisions."""

    diagnostics_l = list(diagnostics or [])
    decisions_l = list(decisions or [])
    prop_summaries: list[dict[str, Any]] = []
    for p in list(proposals or []):
        try:
            pattern, _z_ast, confidence, _extra, meta = p
            prop_summaries.append(
                {
                    "pattern": list(pattern),
                    "confidence": float(confidence),
                    "kind": str((meta or {}).get("kind", "")),
                    "family": str((meta or {}).get("gs_family", "unit_torus")),
                    "gs_kind": str((meta or {}).get("gs_kind", "")),
                    "axes": list((meta or {}).get("gs_axes", [])),
                    "z_human": (meta or {}).get("z_human", None),
                    "pi_exponents": (meta or {}).get("pi_exponents", None),
                    "pi_l1": (meta or {}).get("pi_l1", None),
                }
            )
        except Exception:
            prop_summaries.append({"repr": repr(p)})
    if not diagnostics_l and not prop_summaries and not decisions_l and not context:
        return
    with _LOCK:
        _UNIT_TORUS_EVENTS.append(
            {
                "event_index": len(_UNIT_TORUS_EVENTS),
                "event_type": str(event_type),
                "diagnostics": _jsonable(diagnostics_l),
                "proposals": _jsonable(prop_summaries),
                "decisions": _jsonable(decisions_l),
                "context": _jsonable(context or {}),
            }
        )
def record_de_terms(*, terms: Iterable[Any], context: dict[str, Any] | None = None) -> None:
    """Record DE-library terms injected by the GS layer."""

    terms_s = []
    for term in terms or []:
        try:
            terms_s.append(repr(term))
        except Exception:
            terms_s.append(str(type(term).__name__))
    if not terms_s:
        return
    with _LOCK:
        _DE_EVENTS.append(
            {
                "event_index": len(_DE_EVENTS),
                "terms": terms_s,
                "context": _jsonable(context or {}),
            }
        )


def _flatten_stagea() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for ev in _STAGEA_EVENTS:
        for d in ev.get("diagnostics", []) or []:
            row = dict(d)
            row["event_index"] = ev.get("event_index")
            row["cols"] = list(ev.get("cols", []))
            out.append(row)
    return out


def _dedupe_generators(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    best: dict[tuple, dict[str, Any]] = {}
    for row in rows:
        key = (
            str(row.get("family")),
            str(row.get("kind")),
            tuple(row.get("axes", []) or []),
            str(row.get("invariant")),
            bool(row.get("accepted", False)),
        )
        old = best.get(key)
        if old is None:
            best[key] = dict(row)
            continue
        try:
            if float(row.get("residual_rel", 1.0e99)) < float(old.get("residual_rel", 1.0e99)):
                best[key] = dict(row)
        except Exception:
            pass
    return sorted(
        best.values(),
        key=lambda r: (
            0 if bool(r.get("accepted", False)) else 1,
            str(r.get("family", "")),
            str(r.get("kind", "")),
            float(r.get("residual_rel", 1.0e99) or 1.0e99),
        ),
    )


def build_gs_payload(
    *,
    final_expression: str | None = None,
    mode: str | None = None,
    include_rejected: bool = True,
    top_k_rejected: int = 40,
) -> dict[str, Any]:
    """Build a JSON-serializable GS report payload."""

    with _LOCK:
        rows = _flatten_stagea()
        events = list(_STAGEA_EVENTS)
        jet_events = list(_JET_EVENTS)
        policy_events = list(_POLICY_EVENTS)
        de_events = list(_DE_EVENTS)
        unit_events = list(_UNIT_TORUS_EVENTS)
        metadata = dict(_RUN_METADATA)

    deduped = _dedupe_generators(rows)
    accepted = [r for r in deduped if bool(r.get("accepted", False))]
    rejected = [r for r in deduped if not bool(r.get("accepted", False))]
    rejected = sorted(rejected, key=lambda r: float(r.get("residual_rel", 1.0e99) or 1.0e99))
    rejected_view = rejected[: max(0, int(top_k_rejected))] if include_rejected else []

    proposals: list[dict[str, Any]] = []
    for ev in events:
        for p in ev.get("proposals", []) or []:
            row = dict(p)
            row["event_index"] = ev.get("event_index")
            proposals.append(row)

    fam_counts = Counter(str(r.get("family", "")) for r in accepted)
    kind_counts = Counter(f"{r.get('family')}:{r.get('kind')}" for r in accepted)
    switched_off_counts = Counter(str(r.get("family", "")) for r in rejected)
    jet_prop_count = sum(len(ev.get("proposals", []) or []) for ev in jet_events)

    return {
        "schema": "nestynet_sr_gs_report_v3",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "mode": str(mode or metadata.get("mode") or "unknown"),
        "metadata": _jsonable(metadata),
        "final_expression": final_expression,
        "summary": {
            "stagea_probe_events": len(events),
            "candidate_generators_tested": len(deduped),
            "satisfied_generators": len(accepted),
            "switched_off_generators": len(rejected),
            "quotient_coordinate_proposals": len(proposals),
            "jet_probe_events": len(jet_events),
            "jet_proposals": int(jet_prop_count),
            "policy_events": len(policy_events),
            "de_template_events": len(de_events),
            "unit_torus_events": len(unit_events),
            "satisfied_by_family": dict(fam_counts),
            "satisfied_by_kind": dict(kind_counts),
            "switched_off_by_family": dict(switched_off_counts),
        },
        "satisfied_generators": _jsonable(accepted),
        "quotient_coordinate_proposals": _jsonable(proposals),
        "switched_off_generators_topk": _jsonable(rejected_view),
        "jet_events": _jsonable(jet_events),
        "policy_events": _jsonable(policy_events),
        "de_template_events": _jsonable(de_events),
        "unit_torus_events": _jsonable(unit_events),
        "raw_event_count": len(events),
    }


def format_gs_markdown(payload: dict[str, Any]) -> str:
    s = payload.get("summary", {}) or {}
    lines = [
        "# Generalized Symmetry Report",
        "",
        f"Mode: `{payload.get('mode', 'unknown')}`",
        f"Created UTC: `{payload.get('created_utc', '')}`",
        "",
        "## Final expression",
        "",
        "```text",
        str(payload.get("final_expression") or "<not available>"),
        "```",
        "",
        "## Summary",
        "",
        f"- Stage-A affine-generator probe events: {s.get('stagea_probe_events', 0)}",
        f"- Candidate affine generators tested: {s.get('candidate_generators_tested', 0)}",
        f"- Sample-compatible affine probes: {s.get('satisfied_generators', 0)}",
        f"- Switched-off affine generators: {s.get('switched_off_generators', 0)}",
        f"- Quotient-coordinate proposals emitted: {s.get('quotient_coordinate_proposals', 0)}",
        f"- Jet-level separability probe events: {s.get('jet_probe_events', 0)}",
        f"- Jet-level separability proposals: {s.get('jet_proposals', 0)}",
        f"- DE template events: {s.get('de_template_events', 0)}",
        "",
    ]
    if s.get("satisfied_by_family"):
        lines += ["Sample-compatible probes by family:", ""]
        for k, v in sorted((s.get("satisfied_by_family") or {}).items()):
            lines.append(f"- `{k}`: {v}")
        lines.append("")

    lines += ["## Sample-compatible affine probes", ""]
    acc = payload.get("satisfied_generators", []) or []
    if not acc:
        lines += ["No low-complexity affine generators passed the configured tolerance.", ""]
    else:
        lines += ["| family | kind | axes | invariant / action | residual | heuristic score |", "|---|---|---:|---|---:|---:|"]
        for r in acc[:100]:
            action = str(r.get("invariant") or "")
            alpha = float(r.get("output_alpha") or 0.0)
            beta = float(r.get("output_beta") or 0.0)
            if abs(alpha) > 1e-12 or abs(beta) > 1e-12:
                action += f"; Xf~{alpha:.3g}+{beta:.3g}f"
            lines.append(
                f"| `{r.get('family','')}` | `{r.get('kind','')}` | `{r.get('axes',[])}` | `{action}` | "
                f"{float(r.get('residual_rel', float('nan'))):.3g} | {float(r.get('confidence', 0.0)):.3f} |"
            )
        lines.append("")

    props = payload.get("quotient_coordinate_proposals", []) or []
    lines += ["## Quotient-coordinate proposals", ""]
    if not props:
        lines += ["No strict-invariance witness was converted into a quotient-coordinate proposal.", ""]
    else:
        lines += ["| coordinate | family | kind | axes | pattern | heuristic score |", "|---|---|---|---:|---:|---:|"]
        for r in props[:100]:
            coord = r.get("z_human") or r.get("invariant") or ""
            lines.append(
                f"| `{coord}` | `{r.get('family','')}` | `{r.get('gs_kind','')}` | `{r.get('axes',[])}` | "
                f"`{r.get('pattern',[])}` | {float(r.get('confidence', 0.0)):.3f} |"
            )
        lines.append("")

    rej = payload.get("switched_off_generators_topk", []) or []
    lines += ["## Best rejected affine generators", ""]
    if not rej:
        lines += ["No rejected generators were recorded, or rejected-generator reporting was disabled.", ""]
    else:
        lines += ["| family | kind | axes | invariant | residual | heuristic score |", "|---|---|---:|---|---:|---:|"]
        for r in rej[:100]:
            lines.append(
                f"| `{r.get('family','')}` | `{r.get('kind','')}` | `{r.get('axes',[])}` | `{r.get('invariant','')}` | "
                f"{float(r.get('residual_rel', float('nan'))):.3g} | {float(r.get('confidence', 0.0)):.3f} |"
            )
        lines.append("")

    policy_events = payload.get("policy_events", []) or []
    if policy_events:
        lines += ["## GS policy events", ""]
        lines.append("| policy | action | details |")
        lines.append("|---|---|---|")
        for ev in policy_events[:80]:
            lines.append(f"| `{ev.get('policy','')}` | `{ev.get('action','')}` | `{json.dumps(ev.get('details', {}), sort_keys=True)[:160]}` |")
        lines.append("")

    jet_events = payload.get("jet_events", []) or []
    if jet_events:
        lines += ["## Jet-level separability witnesses", ""]
        for ev in jet_events[:50]:
            ctx = ev.get("context", {}) or {}
            lines.append(f"Event {ev.get('event_index', 0)} policy=`{ctx.get('policy', '')}` replaced_baseline={ctx.get('replaced_baseline', False)}")
            for d in (ev.get("diagnostics", []) or [])[:12]:
                lines.append(
                    f"- `{d.get('family','')}` `{d.get('kind','')}` "
                    f"groups={d.get('group1',[])}|{d.get('group2',[])} "
                    f"metric={d.get('residual_metric', d.get('metric', ''))} accepted={bool(d.get('accepted', False))}"
                )
            lines.append("")

    de_events = payload.get("de_template_events", []) or []
    if de_events:
        lines += ["## GS DE templates injected", ""]
        for ev in de_events:
            lines.append(f"Event {ev.get('event_index', 0)} context=`{ev.get('context', {})}`:")
            for t in ev.get("terms", []) or []:
                lines.append(f"- `{t}`")
            lines.append("")
    unit_events = payload.get("unit_torus_events", []) or []
    if unit_events:
        lines += ["## Unit-torus dimensional GS", ""]
        for ev in unit_events[:80]:
            ctx = ev.get("context", {}) or {}
            lines.append(
                f"Event {ev.get('event_index', 0)} type=`{ev.get('event_type', '')}` "
                f"policy=`{ctx.get('dim_policy', ctx.get('policy', ''))}`"
            )
            for d in (ev.get("diagnostics", []) or [])[:10]:
                lines.append(
                    f"- `{d.get('family','unit_torus')}` `{d.get('kind','')}` "
                    f"accepted={bool(d.get('accepted', False))} "
                    f"invariant=`{d.get('invariant', d.get('candidate', ''))}`"
                )
            for d in (ev.get("decisions", []) or [])[:10]:
                lines.append(
                    f"- decision `{d.get('candidate','')}` baseline={d.get('baseline_accept')} "
                    f"gs={d.get('gs_accept')} final={d.get('final_accept')}"
                )
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def write_gs_reports(
    *,
    json_path: str | os.PathLike[str],
    markdown_path: str | os.PathLike[str] | None = None,
    final_expression: str | None = None,
    mode: str | None = None,
    include_rejected: bool = True,
    top_k_rejected: int = 40,
    append_human_path: str | os.PathLike[str] | None = None,
    report_json_path: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    payload = build_gs_payload(
        final_expression=final_expression,
        mode=mode,
        include_rejected=include_rejected,
        top_k_rejected=top_k_rejected,
    )
    jp = Path(json_path)
    jp.parent.mkdir(parents=True, exist_ok=True)
    jp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    if markdown_path is not None:
        mp = Path(markdown_path)
        mp.parent.mkdir(parents=True, exist_ok=True)
        mp.write_text(format_gs_markdown(payload), encoding="utf-8")
    if append_human_path is not None:
        try:
            hp = Path(append_human_path)
            with hp.open("a", encoding="utf-8") as f:
                f.write("\n\n" + format_gs_markdown(payload))
        except Exception:
            pass
    if report_json_path is not None:
        try:
            rp = Path(report_json_path)
            if rp.exists():
                data = json.loads(rp.read_text(encoding="utf-8"))
                data["generalized_symmetry_report"] = payload
                rp.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
        except Exception:
            pass
    return payload
