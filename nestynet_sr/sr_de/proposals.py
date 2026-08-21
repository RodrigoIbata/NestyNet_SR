# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata

"""DE proposal slate serialization helpers.

This module is intentionally narrow: it gives the current DE engines a common
proposal record without changing how candidates are selected.  The richer
committee, validation, and reservoir logic will build on this representation.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import numpy as np


@dataclass
class DEProposal:
    proposal_id: str
    engine: str
    role_signature: str
    canonical_key: str
    order: int
    x_axis: int
    rhs_payload: dict[str, Any]
    residual_payload: dict[str, Any] | None
    canonical_equation: str
    complexity: float
    pointwise_metrics: dict[str, Any]
    diagnostics: dict[str, Any]
    support: dict[str, Any]
    provenance: dict[str, Any]
    cost: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))


_POINTWISE_KEYS = (
    "rms_train",
    "rms_val",
    "probe_mse",
    "probe_rms",
    "condition_number",
)


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "detach") and callable(getattr(value, "detach")):
        detached = value.detach().cpu()
        if getattr(detached, "ndim", 0) == 0:
            return float(detached.item())
        return detached.tolist()
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return str(value)


def _safe_float(value: Any, default: float | None = None) -> float | None:
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _compact_text(text: Any) -> str:
    return re.sub(r"\s+", "", str(text or "").strip())


def _preprocess_equation_text(text: Any) -> str:
    out = str(text or "").strip()
    out = out.replace("^", "**")

    # Normalize common AST repr fragments before generic derivative aliases.
    out = re.sub(r"\bD2U\([^)]*\)", "d2u", out)
    out = re.sub(r"\bDU\([^)]*\)", "du", out)
    out = re.sub(r"\bU\(\)", "u", out)
    out = re.sub(r"\bVar\(0\)", "x", out)

    # Normalize textual derivative conventions.  Longer tokens must come first.
    replacements = (
        (r"\bu_xx\b", "d2u"),
        (r"\bu''\b", "d2u"),
        (r"\bd2u_dx2\b", "d2u"),
        (r"\bd2u/dx2\b", "d2u"),
        (r"\bu_dot\b", "du"),
        (r"\bu_x\b", "du"),
        (r"\bu'\b", "du"),
        (r"\bdu_dx\b", "du"),
        (r"\bdu/dx\b", "du"),
        (r"\bx0\b", "x"),
    )
    for pattern, repl in replacements:
        out = re.sub(pattern, repl, out)
    return out


def _split_residual_text(text: str) -> str:
    if "=" not in text:
        return text
    lhs, rhs = text.split("=", 1)
    return f"({lhs})-({rhs})"


def _sympy_canonical_residual(text: str) -> str | None:
    try:
        import sympy as sp
    except Exception:
        return None

    x = sp.Symbol("x")
    u = sp.Symbol("u")
    du = sp.Symbol("du")
    d2u = sp.Symbol("d2u")
    local_dict = {
        "x": x,
        "x0": x,
        "u": u,
        "du": du,
        "d2u": d2u,
        "rhs": sp.Symbol("rhs"),
        "sin": sp.sin,
        "cos": sp.cos,
        "tan": sp.tan,
        "asin": sp.asin,
        "acos": sp.acos,
        "atan": sp.atan,
        "exp": sp.exp,
        "log": sp.log,
        "sqrt": sp.sqrt,
        "Abs": sp.Abs,
        "Add": sp.Add,
        "Mul": sp.Mul,
        "Pow": sp.Pow,
    }

    try:
        expr = sp.sympify(_split_residual_text(text), locals=local_dict, evaluate=True)
        try:
            expr = sp.nsimplify(expr, rational=True)
        except Exception:
            pass
        expr = sp.factor_terms(sp.together(expr))
        _, expr = expr.as_content_primitive()
        expanded = sp.expand(expr)
        for deriv_symbol in (d2u, du):
            coeff = sp.expand(expanded.coeff(deriv_symbol))
            if coeff == 0:
                continue
            if coeff.could_extract_minus_sign():
                expr = -expr
            break
        else:
            terms = sorted(sp.Add.make_args(expanded), key=lambda term: sp.sstr(term, order="lex"))
            if terms:
                coeff, _ = terms[0].as_coeff_Mul()
                if coeff.could_extract_minus_sign():
                    expr = -expr
            elif expr.could_extract_minus_sign():
                expr = -expr

        expr = sp.factor_terms(sp.together(expr))
        return _compact_text(sp.sstr(expr, order="lex"))
    except Exception:
        return None


def _flip_additive_signs(text: str) -> str:
    out: list[str] = []
    for ch in text:
        if ch == "+":
            out.append("-")
        elif ch == "-":
            out.append("+")
        else:
            out.append(ch)
    flipped = "".join(out)
    return flipped[1:] if flipped.startswith("+") else flipped


def _fallback_canonical_residual(text: str) -> str:
    residual = _compact_text(_split_residual_text(text))
    residual = re.sub(r"^[+]?(\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\*", "", residual)
    if residual.startswith("-"):
        residual = _flip_additive_signs(residual[1:])
    return residual


def canonicalize_de_equation(equation: Any) -> str:
    """Return a conservative sign/scale-normalized key for a DE equation string."""
    text = _preprocess_equation_text(equation)
    if not text:
        return "residual:<empty>"
    sympy_key = _sympy_canonical_residual(text)
    if sympy_key:
        return f"residual:{sympy_key}"
    return f"residual:{_fallback_canonical_residual(text)}"


def _candidate_equation_text(payload: Mapping[str, Any]) -> str:
    canonical_equations = payload.get("canonical_equations", None)
    if isinstance(canonical_equations, (list, tuple)) and canonical_equations:
        return " ; ".join(str(eq) for eq in canonical_equations if eq is not None)
    for key in ("canonical_equation", "residual_ast", "rhs_ast", "expr_ast"):
        value = payload.get(key, None)
        if value not in (None, ""):
            return str(value)
    return json.dumps(_jsonable(payload), sort_keys=True)


def _canonical_key_for_payload(payload: Mapping[str, Any]) -> str:
    canonical_equations = payload.get("canonical_equations", None)
    if isinstance(canonical_equations, (list, tuple)) and canonical_equations:
        keys = [canonicalize_de_equation(eq) for eq in canonical_equations if eq not in (None, "")]
        if keys:
            return "multi:" + "|".join(keys)
    return canonicalize_de_equation(_candidate_equation_text(payload))


def _role_signature(payload: Mapping[str, Any], *, default_engine: str) -> str:
    kind = str(payload.get("kind", "") or "").strip()
    engine = str(payload.get("engine", default_engine) or default_engine).strip()
    lane = str(payload.get("lane", "") or "").strip()
    if lane:
        family = str(payload.get("family", "") or "").strip()
        return ":".join(part for part in ("typed", lane, family) if part)
    if engine == "factorized_search":
        mapping_kind = str(payload.get("mapping_kind", "") or "").strip()
        return ":".join(part for part in ("whole_rhs_fss", mapping_kind) if part)
    if engine == "factorized" or kind == "factorized_blocks":
        return "typed_factorized"
    if kind:
        return kind
    return engine


def _candidate_complexity(payload: Mapping[str, Any]) -> float:
    for key in ("complexity", "num_terms", "size", "mapping_complexity"):
        value = _safe_float(payload.get(key, None), None)
        if value is not None:
            return float(value)
    terms = payload.get("terms", None)
    if isinstance(terms, (list, tuple)):
        return float(len(terms))
    equation = _candidate_equation_text(payload)
    return float(max(1, len(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", equation))))


def _pointwise_metrics(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {key: _jsonable(payload.get(key, None)) for key in _POINTWISE_KEYS}


def _residual_payload(payload: Mapping[str, Any]) -> dict[str, Any] | None:
    out = {}
    for key in ("residual_ast", "residual_asts", "canonical_equation", "canonical_equations"):
        if payload.get(key, None) is not None:
            out[key] = payload.get(key)
    return _jsonable(out) if out else None


def _proposal_id(source_id: str, engine: str, canonical_key: str, payload: Mapping[str, Any]) -> str:
    rank = payload.get("candidate_rank", payload.get("shortlist_rank", ""))
    digest_payload = {
        "source_id": source_id,
        "engine": engine,
        "canonical_key": canonical_key,
        "rank": rank,
        "role_signature": _role_signature(payload, default_engine=engine),
    }
    digest = hashlib.sha1(
        json.dumps(_jsonable(digest_payload), sort_keys=True).encode("utf-8")
    ).hexdigest()[:12]
    source_slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", source_id).strip("_") or "proposal"
    return f"{source_slug}:{digest}"


def _proposal_from_payload(
    payload: Mapping[str, Any] | None,
    *,
    source_id: str,
    default_engine: str,
    selected: bool = False,
) -> DEProposal | None:
    if not isinstance(payload, Mapping):
        return None
    clean_payload = _jsonable(copy.deepcopy(dict(payload)))
    engine = str(clean_payload.get("engine", default_engine) or default_engine)
    role_signature = _role_signature(clean_payload, default_engine=engine)
    canonical_key = _canonical_key_for_payload(clean_payload)
    proposal_id = _proposal_id(source_id, engine, canonical_key, clean_payload)
    order = _safe_int(clean_payload.get("order", 0), 0)
    x_axis = _safe_int(clean_payload.get("x_axis", 0), 0)
    diagnostics = clean_payload.get("diagnostics", {})
    if not isinstance(diagnostics, dict):
        diagnostics = {"raw": diagnostics}

    support = {
        "support_count": 1,
        "sources": [source_id],
        "engines": [engine],
        "selected": bool(selected),
    }
    provenance = {
        "source_id": source_id,
        "selected": bool(selected),
        "candidate_rank": clean_payload.get("candidate_rank", None),
        "shortlist_rank": clean_payload.get("shortlist_rank", None),
        "internal_selected_shortlist_rank": clean_payload.get("internal_selected_shortlist_rank", None),
    }
    cost = {}
    for key in ("solve_time_s", "wall_time_s", "walltime_s", "runtime_s"):
        value = _safe_float(clean_payload.get(key, None), None)
        if value is not None:
            cost[key] = value

    return DEProposal(
        proposal_id=proposal_id,
        engine=engine,
        role_signature=role_signature,
        canonical_key=canonical_key,
        order=order,
        x_axis=x_axis,
        rhs_payload=clean_payload,
        residual_payload=_residual_payload(clean_payload),
        canonical_equation=str(clean_payload.get("canonical_equation", "") or ""),
        complexity=_candidate_complexity(clean_payload),
        pointwise_metrics=_pointwise_metrics(clean_payload),
        diagnostics=_jsonable(diagnostics),
        support=support,
        provenance=provenance,
        cost=_jsonable(cost),
    )


def proposal_from_stlsq_result(
    payload: Mapping[str, Any] | None,
    *,
    source_id: str = "first_line",
    selected: bool = False,
) -> DEProposal | None:
    return _proposal_from_payload(
        payload,
        source_id=source_id,
        default_engine="stlsq",
        selected=selected,
    )


def proposal_from_factorized_result(
    payload: Mapping[str, Any] | None,
    *,
    source_id: str = "factorized_rescue",
    selected: bool = False,
) -> DEProposal | None:
    return _proposal_from_payload(
        payload,
        source_id=source_id,
        default_engine="factorized",
        selected=selected,
    )


def proposal_from_factorized_search_result(
    payload: Mapping[str, Any] | None,
    *,
    source_id: str = "factorized_search_rescue",
    selected: bool = False,
) -> DEProposal | None:
    return _proposal_from_payload(
        payload,
        source_id=source_id,
        default_engine="factorized_search",
        selected=selected,
    )


def _merge_support(base: DEProposal, incoming: DEProposal) -> None:
    sources = list(base.support.get("sources", []) or [])
    engines = list(base.support.get("engines", []) or [])
    for source in incoming.support.get("sources", []) or []:
        if source not in sources:
            sources.append(source)
    for engine in incoming.support.get("engines", []) or []:
        if engine not in engines:
            engines.append(engine)
    base.support["sources"] = sources
    base.support["engines"] = engines
    base.support["support_count"] = len(sources)
    base.support["selected"] = bool(base.support.get("selected", False) or incoming.support.get("selected", False))

    merged = list(base.support.get("merged_proposals", []) or [])
    merged.append(
        {
            "proposal_id": incoming.proposal_id,
            "engine": incoming.engine,
            "source_id": incoming.provenance.get("source_id"),
            "candidate_rank": incoming.provenance.get("candidate_rank"),
            "shortlist_rank": incoming.provenance.get("shortlist_rank"),
            "selected": bool(incoming.provenance.get("selected", False)),
        }
    )
    base.support["merged_proposals"] = merged


def _unique_strings(values: Sequence[Any]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value)
        if text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _namespace_source(value: Any, namespace: str | None) -> str:
    text = str(value)
    if not namespace:
        return text
    prefix = f"{namespace}:"
    return text if text.startswith(prefix) else f"{prefix}{text}"


def _proposal_dict_key(proposal: Mapping[str, Any]) -> tuple[str, int, int]:
    canonical_key = str(proposal.get("canonical_key", "") or "")
    if not canonical_key:
        canonical_key = _canonical_key_for_payload(proposal.get("rhs_payload", proposal))
    return (
        canonical_key,
        _safe_int(proposal.get("order", 0), 0),
        _safe_int(proposal.get("x_axis", 0), 0),
    )


def _namespace_proposal_dict(
    proposal: Mapping[str, Any],
    *,
    namespace: str | None,
) -> dict[str, Any]:
    out = _jsonable(copy.deepcopy(dict(proposal)))
    if not namespace:
        return out
    namespace_s = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(namespace)).strip("_")
    if not namespace_s:
        return out

    pid = str(out.get("proposal_id", "") or "")
    if pid:
        out["proposal_id"] = _namespace_source(pid, namespace_s)

    support = dict(out.get("support", {}) or {})
    sources = support.get("sources", None)
    if not isinstance(sources, list) or not sources:
        source_id = (out.get("provenance", {}) or {}).get("source_id", pid or namespace_s)
        sources = [source_id]
    support["sources"] = _unique_strings(_namespace_source(source, namespace_s) for source in sources)
    namespaces = list(support.get("source_namespaces", []) or [])
    namespaces.append(namespace_s)
    support["source_namespaces"] = _unique_strings(namespaces)
    support["support_count"] = len(support["sources"])
    out["support"] = support

    provenance = dict(out.get("provenance", {}) or {})
    if provenance.get("source_id", None) is not None:
        provenance["source_id"] = _namespace_source(provenance["source_id"], namespace_s)
    else:
        provenance["source_id"] = namespace_s
    provenance["source_namespace"] = namespace_s
    out["provenance"] = provenance
    return out


def _merge_proposal_dict_support(base: dict[str, Any], incoming: Mapping[str, Any]) -> None:
    base_support = dict(base.get("support", {}) or {})
    incoming_support = dict(incoming.get("support", {}) or {})

    sources = _unique_strings(
        list(base_support.get("sources", []) or [])
        + list(incoming_support.get("sources", []) or [])
    )
    engines = _unique_strings(
        list(base_support.get("engines", []) or [])
        + list(incoming_support.get("engines", []) or [])
    )
    namespaces = _unique_strings(
        list(base_support.get("source_namespaces", []) or [])
        + list(incoming_support.get("source_namespaces", []) or [])
    )
    base_support["sources"] = sources
    base_support["engines"] = engines
    base_support["support_count"] = len(sources)
    base_support["selected"] = bool(
        base_support.get("selected", False) or incoming_support.get("selected", False)
    )
    if namespaces:
        base_support["source_namespaces"] = namespaces

    merged = list(base_support.get("merged_proposals", []) or [])
    incoming_id = str(incoming.get("proposal_id", "") or "")
    seen_ids = {str(row.get("proposal_id", "") or "") for row in merged if isinstance(row, Mapping)}
    if incoming_id and incoming_id not in seen_ids and incoming_id != str(base.get("proposal_id", "") or ""):
        merged.append(
            {
                "proposal_id": incoming_id,
                "engine": incoming.get("engine", None),
                "source_id": (incoming.get("provenance", {}) or {}).get("source_id", None),
                "candidate_rank": (incoming.get("provenance", {}) or {}).get("candidate_rank", None),
                "shortlist_rank": (incoming.get("provenance", {}) or {}).get("shortlist_rank", None),
                "selected": bool((incoming.get("provenance", {}) or {}).get("selected", False)),
            }
        )
    if merged:
        base_support["merged_proposals"] = merged
    base["support"] = _jsonable(base_support)


def merge_proposal_slates(
    slates: Sequence[Sequence[Mapping[str, Any]] | None],
    *,
    source_namespaces: Sequence[str | None] | None = None,
) -> list[dict[str, Any]]:
    """Merge proposal slates into a DE proposal reservoir.

    The reservoir key is conservative: canonical equation key, order, and
    x-axis.  Support counts unique namespaced sources, so duplicate reports
    from the same source do not inflate support.
    """

    namespaces = list(source_namespaces or [])
    out: list[dict[str, Any]] = []
    seen: dict[tuple[str, int, int], int] = {}
    for slate_idx, slate in enumerate(list(slates or [])):
        namespace = namespaces[slate_idx] if slate_idx < len(namespaces) else None
        for proposal in list(slate or []):
            if not isinstance(proposal, Mapping):
                continue
            clean = _namespace_proposal_dict(proposal, namespace=namespace)
            key = _proposal_dict_key(clean)
            prior = seen.get(key, None)
            if prior is None:
                seen[key] = len(out)
                out.append(clean)
            else:
                _merge_proposal_dict_support(out[prior], clean)
    return [_jsonable(row) for row in out]


def _dedupe_key(proposal: DEProposal) -> tuple[str, int, int]:
    return (proposal.canonical_key, int(proposal.order), int(proposal.x_axis))


def _append_proposal(
    proposals: list[DEProposal],
    seen: dict[tuple[str, int, int], int],
    proposal: DEProposal | None,
) -> None:
    if proposal is None:
        return
    key = _dedupe_key(proposal)
    prior = seen.get(key, None)
    if prior is None:
        seen[key] = len(proposals)
        proposals.append(proposal)
        return
    _merge_support(proposals[prior], proposal)


def _iter_shortlist(parent: Mapping[str, Any] | None) -> list[Mapping[str, Any]]:
    if not isinstance(parent, Mapping):
        return []
    shortlist = parent.get("shortlist", None)
    if not isinstance(shortlist, list):
        return []
    return [row for row in shortlist if isinstance(row, Mapping)]


def build_proposal_slate(
    *,
    first_line: Mapping[str, Any] | None = None,
    factorized: Mapping[str, Any] | None = None,
    factorized_search: Mapping[str, Any] | None = None,
    selected: Mapping[str, Any] | None = None,
    selected_engine: str | None = None,
) -> list[dict[str, Any]]:
    """Build an additive proposal slate from the current serialized DE report payloads."""
    proposals: list[DEProposal] = []
    seen: dict[tuple[str, int, int], int] = {}

    _append_proposal(
        proposals,
        seen,
        proposal_from_stlsq_result(first_line, source_id="first_line"),
    )
    _append_proposal(
        proposals,
        seen,
        proposal_from_factorized_result(factorized, source_id="factorized_rescue"),
    )
    for idx, row in enumerate(_iter_shortlist(factorized)):
        _append_proposal(
            proposals,
            seen,
            proposal_from_factorized_result(
                row,
                source_id=f"factorized_rescue.shortlist.{idx}",
            ),
        )

    _append_proposal(
        proposals,
        seen,
        proposal_from_factorized_search_result(
            factorized_search,
            source_id="factorized_search_rescue",
        ),
    )
    for idx, row in enumerate(_iter_shortlist(factorized_search)):
        _append_proposal(
            proposals,
            seen,
            proposal_from_factorized_search_result(
                row,
                source_id=f"factorized_search_rescue.shortlist.{idx}",
            ),
        )

    if selected is not None:
        engine = str(selected_engine or selected.get("engine", "") or "").strip()
        if engine == "factorized":
            selected_proposal = proposal_from_factorized_result(
                selected,
                source_id="selected",
                selected=True,
            )
        elif engine == "factorized_search":
            selected_proposal = proposal_from_factorized_search_result(
                selected,
                source_id="selected",
                selected=True,
            )
        else:
            selected_proposal = proposal_from_stlsq_result(
                selected,
                source_id="selected",
                selected=True,
            )
        _append_proposal(proposals, seen, selected_proposal)

    return [proposal.to_dict() for proposal in proposals]


__all__ = [
    "DEProposal",
    "build_proposal_slate",
    "canonicalize_de_equation",
    "merge_proposal_slates",
    "proposal_from_factorized_result",
    "proposal_from_factorized_search_result",
    "proposal_from_stlsq_result",
]
