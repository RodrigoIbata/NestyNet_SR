# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""Versioned coefficient identity/value/unit metadata.

Symbolic expressions must keep named physical constants as symbols: replacing a
unitful ``c`` by the bare number ``2.5`` destroys its dimensional identity.  The
records in this module carry that identity alongside the fitted numeric value so
symbolic consumers can retain ``c`` for display and unit checking while
substituting ``c = 2.5`` only for numerical evaluation.

Dimensionless and dimensionful coefficients use the same record format.  A
dimensionless coefficient simply has an all-zero exact dimension vector.
"""

from __future__ import annotations

import ast
import keyword
import math
import operator
import re
from fractions import Fraction
from typing import Any, Mapping, Optional, Sequence

from .bridges import AtomNode, Node, collect_all_atoms
from .units import normalize_free_const_scope


COEFFICIENT_METADATA_SCHEMA = "coefficient_metadata_v1"

_FREE_CONST_KINDS = {"free_const", "freeconst", "free_constant"}
_FIXED_CONST_KINDS = {"fixed_const", "fixedconst", "fixed_constant"}
_SCALE_KINDS = {"scale", "mul_scale"}
_SYMBOL_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_COEFFICIENT_SYMBOL_ESCAPE_PREFIX = "coef_"
_SAFE_EXPRESSION_CALLS = frozenset(
    {
        "Abs",
        "abs",
        "acos",
        "acosh",
        "arccos",
        "arccosh",
        "arcsin",
        "arcsinh",
        "arctan",
        "arctanh",
        "asin",
        "asinh",
        "atan",
        "atan2",
        "atanh",
        "ceiling",
        "cos",
        "cosh",
        "erf",
        "exp",
        "floor",
        "ln",
        "log",
        "sign",
        "sin",
        "sinh",
        "sqrt",
        "tan",
        "tanh",
    }
)
_SAFE_EXPRESSION_BINOPS = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow)
_SAFE_EXPRESSION_UNARYOPS = (ast.UAdd, ast.USub)
_MAX_EXPRESSION_TEXT = 65_536
_MAX_EXPRESSION_NODES = 4_096
_MAX_EXPRESSION_DEPTH = 256
_MAX_EXPRESSION_INTEGER_BITS = 256
_MAX_EXPRESSION_CONSTANT_BITS = 1_024
_MAX_EXPRESSION_EXPONENT = Fraction(1_024, 1)


def _validated_math_expression_text(expression: str) -> tuple[str, set[str]]:
    """Validate the non-executing grammar accepted at report boundaries."""

    normalized = str(expression).replace("^", "**")
    if not normalized.strip():
        raise CoefficientMetadataError(
            "coefficient_expression_parse_failed",
            "coefficient expression is empty",
        )
    if len(normalized) > _MAX_EXPRESSION_TEXT:
        raise CoefficientMetadataError(
            "coefficient_expression_unsafe",
            "coefficient expression exceeds the safe text-size limit",
        )
    try:
        tree = ast.parse(normalized, mode="eval")
    except (MemoryError, RecursionError, SyntaxError, ValueError) as exc:
        raise CoefficientMetadataError(
            "coefficient_expression_parse_failed",
            f"could not parse coefficient expression syntax: {exc}",
        ) from exc

    symbol_names: set[str] = set()
    node_count = 0

    def reject(reason: str) -> None:
        raise CoefficientMetadataError(
            "coefficient_expression_unsafe",
            f"unsafe coefficient expression syntax: {reason}",
        )

    def numeric_value(node: ast.AST) -> Fraction | None:
        if isinstance(node, ast.Constant):
            value = node.value
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                return None
            return Fraction(str(value))
        if isinstance(node, ast.UnaryOp) and isinstance(
            node.op, _SAFE_EXPRESSION_UNARYOPS
        ):
            value = numeric_value(node.operand)
            if value is None:
                return None
            return value if isinstance(node.op, ast.UAdd) else -value
        if isinstance(node, ast.BinOp) and isinstance(
            node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)
        ):
            left = numeric_value(node.left)
            right = numeric_value(node.right)
            if left is None or right is None:
                return None
            if isinstance(node.op, ast.Add):
                value = left + right
            elif isinstance(node.op, ast.Sub):
                value = left - right
            elif isinstance(node.op, ast.Mult):
                value = left * right
            else:
                if right == 0:
                    reject("numeric exponent divides by zero")
                value = left / right
            if (
                value.numerator.bit_length() > _MAX_EXPRESSION_CONSTANT_BITS
                or value.denominator.bit_length() > _MAX_EXPRESSION_CONSTANT_BITS
            ):
                reject("numeric exponent exceeds the safe precision limit")
            return value
        return None

    def visit(node: ast.AST, depth: int) -> None:
        nonlocal node_count
        node_count += 1
        if node_count > _MAX_EXPRESSION_NODES:
            reject("expression exceeds the safe node-count limit")
        if depth > _MAX_EXPRESSION_DEPTH:
            reject("expression exceeds the safe nesting-depth limit")
        if isinstance(node, ast.Expression):
            visit(node.body, depth + 1)
            return
        if isinstance(node, ast.BinOp):
            if not isinstance(node.op, _SAFE_EXPRESSION_BINOPS):
                reject(f"operator {type(node.op).__name__} is not allowed")
            visit(node.left, depth + 1)
            visit(node.right, depth + 1)
            if isinstance(node.op, ast.Pow):
                exponent = numeric_value(node.right)
                has_symbolic_exponent = any(
                    isinstance(child, (ast.Call, ast.Name))
                    for child in ast.walk(node.right)
                )
                if exponent is None and not has_symbolic_exponent:
                    reject("numeric exponent syntax is not safely bounded")
                if (
                    exponent is not None
                    and abs(exponent) > _MAX_EXPRESSION_EXPONENT
                ):
                    reject("numeric exponent exceeds the safe magnitude limit")
            return
        if isinstance(node, ast.UnaryOp):
            if not isinstance(node.op, _SAFE_EXPRESSION_UNARYOPS):
                reject(f"operator {type(node.op).__name__} is not allowed")
            visit(node.operand, depth + 1)
            return
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                reject("only direct calls to approved mathematical functions are allowed")
            if node.func.id not in _SAFE_EXPRESSION_CALLS:
                reject(f"function {node.func.id!r} is not allowed")
            if node.keywords:
                reject("keyword arguments are not allowed")
            for argument in node.args:
                visit(argument, depth + 1)
            return
        if isinstance(node, ast.Name):
            if not _SYMBOL_RE.fullmatch(node.id) or node.id.startswith("__"):
                reject(f"symbol {node.id!r} is not allowed")
            symbol_names.add(node.id)
            return
        if isinstance(node, ast.Constant):
            value = node.value
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                reject("only real numeric literals are allowed")
            if isinstance(value, float) and not math.isfinite(value):
                reject("numeric literals must be finite")
            if (
                isinstance(value, int)
                and value.bit_length() > _MAX_EXPRESSION_INTEGER_BITS
            ):
                reject("integer literal exceeds the safe precision limit")
            return
        reject(f"node {type(node).__name__} is not allowed")

    try:
        visit(tree, 0)
    except RecursionError as exc:
        raise CoefficientMetadataError(
            "coefficient_expression_unsafe",
            "coefficient expression exceeds the safe recursion limit",
        ) from exc
    return normalized, symbol_names


class CoefficientMetadataError(ValueError):
    """A stable, fail-closed coefficient-metadata validation error."""

    def __init__(self, code: str, reason: str):
        super().__init__(reason)
        self.code = str(code)
        self.reason = str(reason)


def _canonical_scalar_kind(kind: Any) -> Optional[str]:
    value = str(kind or "").strip().lower()
    if value in _FREE_CONST_KINDS:
        return "free_const"
    if value in _FIXED_CONST_KINDS:
        return "fixed_const"
    if value in _SCALE_KINDS:
        return "scale"
    return None


def _fraction(value: Any) -> Fraction:
    if isinstance(value, Fraction):
        return value
    if isinstance(value, bool):
        raise TypeError("boolean dimension exponents are invalid")
    if hasattr(value, "item"):
        try:
            value = value.item()
        except Exception:
            pass
    if isinstance(value, int):
        return Fraction(value, 1)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("dimension exponents must be finite")
        return Fraction.from_float(value).limit_denominator(1024)
    return Fraction(str(value))


def _dimension_payload(dimension: Sequence[Any]) -> list[str]:
    if isinstance(dimension, (str, bytes, Mapping)):
        raise TypeError("dimension must be a sequence of exact exponents")
    return [str(_fraction(value)) for value in dimension]


def _finite_float(value: Any, *, label: str) -> float:
    if isinstance(value, bool):
        raise CoefficientMetadataError(
            "coefficient_value_invalid", f"{label} must not be boolean"
        )
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "numel"):
        if int(value.numel()) != 1:
            raise CoefficientMetadataError(
                "coefficient_value_not_scalar",
                f"{label} contains {int(value.numel())} values; expected one scalar",
            )
        value = value.reshape(-1)[0].item()
    elif hasattr(value, "item"):
        value = value.item()
    try:
        out = float(value)
    except Exception as exc:
        raise CoefficientMetadataError(
            "coefficient_value_invalid", f"{label} is not numeric: {value!r}"
        ) from exc
    if not math.isfinite(out):
        raise CoefficientMetadataError(
            "coefficient_value_nonfinite", f"{label} is non-finite: {out!r}"
        )
    return out


def _is_plain_sympy_symbol_name(name: str) -> bool:
    """Return whether bare ``name`` parses as that exact SymPy Symbol.

    SymPy's public namespace is much larger than a practical hand-maintained
    reserved-name list (``gamma``, ``N``, ``S``, ``Q``, ...).  Python keywords
    and literals such as ``None`` also cannot safely be used as parser locals.
    Probe the parser itself so new SymPy globals fail safely as well.
    """

    if keyword.iskeyword(name):
        return False
    try:
        import sympy as sp

        parsed = sp.sympify(name, evaluate=False)
    except Exception:
        return False
    return isinstance(parsed, sp.Symbol) and str(parsed) == name


def validate_coefficient_symbol(name: Any) -> str:
    """Return a safe symbolic coefficient name or raise a stable error."""

    symbol = str(name or "").strip()
    if not symbol:
        raise CoefficientMetadataError(
            "coefficient_symbol_missing", "named coefficient has no symbol"
        )
    if _SYMBOL_RE.fullmatch(symbol) is None:
        raise CoefficientMetadataError(
            "coefficient_symbol_invalid",
            f"coefficient symbol {symbol!r} is not a simple identifier",
        )
    if not _is_plain_sympy_symbol_name(symbol):
        raise CoefficientMetadataError(
            "coefficient_symbol_reserved",
            f"coefficient symbol {symbol!r} collides with a reserved SymPy name",
        )
    return symbol


def validate_coefficient_name(name: Any) -> str:
    """Return a valid logical coefficient name.

    Logical names may coincide with SymPy built-ins (for example ``pi``).
    Those names are rendered through :func:`coefficient_symbol_for_name` so
    the configured value is never silently replaced by a SymPy constant.
    """

    logical_name = str(name or "").strip()
    if not logical_name:
        raise CoefficientMetadataError(
            "coefficient_name_missing", "named coefficient has no logical name"
        )
    if _SYMBOL_RE.fullmatch(logical_name) is None:
        raise CoefficientMetadataError(
            "coefficient_name_invalid",
            f"coefficient name {logical_name!r} is not a simple identifier",
        )
    return logical_name


def coefficient_symbol_for_name(name: Any) -> str:
    """Map a logical coefficient name to a safe, deterministic SymPy symbol."""

    logical_name = validate_coefficient_name(name)
    # Escape both unsafe parser names and the escape namespace itself.  The
    # latter makes the map injective: pi -> coef_pi, while a logical coef_pi
    # becomes coef_coef_pi rather than colliding with pi.
    symbol = logical_name
    if (
        logical_name.startswith(_COEFFICIENT_SYMBOL_ESCAPE_PREFIX)
        or re.fullmatch(r"x\d+", logical_name) is not None
        or not _is_plain_sympy_symbol_name(logical_name)
    ):
        symbol = f"{_COEFFICIENT_SYMBOL_ESCAPE_PREFIX}{logical_name}"
    return validate_coefficient_symbol(symbol)


def named_coefficient_symbol(atom: AtomNode) -> Optional[str]:
    """Return the symbolic name for a FreeConst/FixedConst atom.

    ``Scale`` atoms intentionally return ``None``: they remain anonymous
    dimensionless numeric coefficients in printed expressions.
    """

    kind = _canonical_scalar_kind(getattr(atom, "kind", None))
    if kind not in {"free_const", "fixed_const"}:
        return None
    kwargs = getattr(atom, "kwargs", None) or {}
    raw_name = kwargs.get("name", None)
    if raw_name is None:
        raw_name = getattr(atom, "tag", None)
    return coefficient_symbol_for_name(raw_name)


def _strict_nonnegative_int(
    value: Any,
    *,
    code: str,
    label: str,
    allow_none: bool = False,
) -> Optional[int]:
    if value is None and allow_none:
        return None
    if isinstance(value, bool):
        raise CoefficientMetadataError(code, f"{label} must be a non-negative integer")
    try:
        normalized = int(operator.index(value))
    except Exception as exc:
        raise CoefficientMetadataError(
            code, f"{label} must be a non-negative integer"
        ) from exc
    if normalized < 0:
        raise CoefficientMetadataError(code, f"{label} must be non-negative")
    return normalized


def empty_coefficient_metadata(
    *,
    dimension_basis: Sequence[Any] = (),
    source: str = "unspecified",
    dataset_id: Optional[str] = None,
    dataset_index: Optional[int] = None,
) -> dict[str, Any]:
    """Return a valid empty v1 metadata bundle."""

    return {
        "schema": COEFFICIENT_METADATA_SCHEMA,
        "valid": True,
        "code": "coefficient_metadata_ok",
        "reason": "coefficient metadata is valid",
        "source": str(source),
        "dimension_basis": [str(value) for value in dimension_basis],
        "dataset_id": None if dataset_id is None else str(dataset_id),
        "dataset_index": None if dataset_index is None else int(dataset_index),
        "record_count": 0,
        "symbol_count": 0,
        "records": [],
    }


def _invalid_metadata(
    code: str,
    reason: str,
    *,
    dimension_basis: Sequence[Any] = (),
    source: str = "ast_scalar_coefficients_v1",
    records: Sequence[Mapping[str, Any]] = (),
    dataset_id: Optional[str] = None,
    dataset_index: Optional[int] = None,
) -> dict[str, Any]:
    return {
        "schema": COEFFICIENT_METADATA_SCHEMA,
        "valid": False,
        "code": str(code),
        "reason": str(reason),
        "source": str(source),
        "dimension_basis": [str(value) for value in dimension_basis],
        "dataset_id": None if dataset_id is None else str(dataset_id),
        "dataset_index": None if dataset_index is None else int(dataset_index),
        "record_count": len(records),
        "symbol_count": len(
            {str(record.get("symbol")) for record in records if record.get("symbol")}
        ),
        "records": [dict(record) for record in records],
    }


def _unwrap_leaf(leaf: Any) -> Any:
    current = leaf
    for _ in range(8):
        nxt = None
        for attr in ("core", "model", "base_model"):
            try:
                candidate = getattr(current, attr, None)
            except Exception:
                candidate = None
            if candidate is not None and candidate is not current:
                nxt = candidate
                break
        if nxt is None:
            break
        current = nxt
    return current


def _record_semantics(record: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        record.get("kind"),
        record.get("symbol"),
        record.get("scope"),
        tuple(record.get("dimension") or ()),
        record.get("dimension_status"),
        bool(record.get("trainable")),
        record.get("display"),
    )


def collect_coefficient_metadata(
    root: Node,
    model: Any,
    units_spec: Any = None,
    *,
    dataset_id: Optional[str] = None,
    dataset_index: Optional[int] = None,
) -> dict[str, Any]:
    """Collect named constants and anonymous Scale leaves from one fitted AST.

    Repeated occurrences of the same tagged coefficient are collapsed into one
    logical record with multiple ``occurrences``.  Conflicting values, scopes,
    kinds, or dimensions fail closed instead of silently choosing one.
    """

    source = "ast_scalar_coefficients_v1"
    basis = tuple(
        str(value)
        for value in getattr(getattr(units_spec, "unit_system", None), "base", ())
    )
    records: list[dict[str, Any]] = []
    by_identity: dict[str, dict[str, Any]] = {}
    identity_topology: dict[str, tuple[Optional[str], int]] = {}
    tag_owner: dict[str, str] = {}
    core_owner: dict[int, str] = {}
    try:
        atoms = list(collect_all_atoms(root))
        if not any(
            _canonical_scalar_kind(getattr(atom, "kind", None)) is not None
            for atom in atoms
        ):
            return normalize_coefficient_metadata(
                empty_coefficient_metadata(
                    dimension_basis=basis,
                    source=source,
                    dataset_id=dataset_id,
                    dataset_index=dataset_index,
                ),
                require_values=True,
                units_spec=units_spec,
            )
        leaves_raw = getattr(model, "leaf", None)
        if leaves_raw is None:
            raise CoefficientMetadataError(
                "coefficient_leaf_mapping_unavailable",
                "fitted model has no .leaf collection",
            )
        leaves = list(leaves_raw)
        if len(atoms) != len(leaves):
            raise CoefficientMetadataError(
                "coefficient_leaf_mapping_mismatch",
                f"AST has {len(atoms)} atoms but fitted model has {len(leaves)} leaves",
            )

        for atom_index, (atom, leaf) in enumerate(zip(atoms, leaves)):
            kind = _canonical_scalar_kind(getattr(atom, "kind", None))
            if kind is None:
                continue
            core = _unwrap_leaf(leaf)
            if not hasattr(core, "value"):
                raise CoefficientMetadataError(
                    "coefficient_value_unavailable",
                    f"{kind} atom at index {atom_index} has no scalar value",
                )
            value = _finite_float(
                getattr(core, "value"), label=f"{kind} atom at index {atom_index}"
            )
            kwargs = getattr(atom, "kwargs", None) or {}
            symbol = named_coefficient_symbol(atom)
            tag_raw = getattr(atom, "tag", None)
            atom_tag = None if tag_raw is None else str(tag_raw)
            name_raw = kwargs.get("name", None)
            if name_raw is None:
                name_raw = atom_tag
            name = None if name_raw is None else validate_coefficient_name(name_raw)

            if kind == "fixed_const":
                scope = "fixed"
                trainable = False
            else:
                scope = normalize_free_const_scope(
                    getattr(atom, "scope", "experiment"), default="experiment"
                )
                trainable = True

            dimension = None
            dimension_status = "unavailable"
            if units_spec is not None:
                if kind == "free_const":
                    declared_dims = dict(
                        getattr(units_spec, "free_const_dims", {}) or {}
                    )
                    if name not in declared_dims:
                        raise CoefficientMetadataError(
                            "coefficient_dimension_undeclared",
                            f"free coefficient {name!r} has no declared dimension",
                        )
                    dimension = _dimension_payload(declared_dims[name])
                    dimension_status = "declared"
                    declared_scopes = dict(
                        getattr(units_spec, "free_const_scope", {}) or {}
                    )
                    declared_scope = normalize_free_const_scope(
                        declared_scopes.get(name, "experiment"),
                        default="experiment",
                    )
                    if scope != declared_scope:
                        raise CoefficientMetadataError(
                            "coefficient_scope_conflict",
                            f"free coefficient {name!r} has AST scope {scope!r} "
                            f"but declared scope {declared_scope!r}",
                        )
                elif kind == "fixed_const":
                    declared_dims = dict(
                        getattr(units_spec, "fixed_const_dims", {}) or {}
                    )
                    if name not in declared_dims:
                        raise CoefficientMetadataError(
                            "coefficient_dimension_undeclared",
                            f"fixed coefficient {name!r} has no declared dimension",
                        )
                    dimension = _dimension_payload(declared_dims[name])
                    dimension_status = "declared"
                    declared_values = dict(
                        getattr(units_spec, "fixed_const_values", {}) or {}
                    )
                    if name in declared_values:
                        declared_value = _finite_float(
                            declared_values[name],
                            label=f"declared fixed coefficient {name!r}",
                        )
                        if not math.isclose(
                            value, declared_value, rel_tol=1.0e-12, abs_tol=1.0e-15
                        ):
                            raise CoefficientMetadataError(
                                "fixed_coefficient_value_conflict",
                                f"fixed coefficient {name!r} has fitted value {value!r} "
                                f"but declared value {declared_value!r}",
                            )
                else:
                    dimension = ["0" for _ in basis]
                    dimension_status = "dimensionless"

            if dimension is not None and len(dimension) != len(basis):
                raise CoefficientMetadataError(
                    "coefficient_dimension_rank_mismatch",
                    f"{kind} coefficient {symbol or name or atom_index!r} has dimension "
                    f"rank {len(dimension)}; expected {len(basis)}",
                )

            if symbol is not None:
                identity = f"{kind}:{name}"
                display = "symbol"
            else:
                identity_suffix = atom_tag if atom_tag else str(atom_index)
                identity = f"scale:{identity_suffix}"
                display = "numeric"

            topology = (atom_tag, id(core))
            prior_topology = identity_topology.get(identity)
            if prior_topology is not None and prior_topology != topology:
                raise CoefficientMetadataError(
                    "coefficient_identity_topology_conflict",
                    f"coefficient identity {identity!r} refers to more than one "
                    "tagged fitted parameter",
                )
            identity_topology[identity] = topology
            if atom_tag is not None:
                prior_tag_owner = tag_owner.get(atom_tag)
                if prior_tag_owner is not None and prior_tag_owner != identity:
                    raise CoefficientMetadataError(
                        "coefficient_parameter_alias_conflict",
                        f"coefficient tag {atom_tag!r} is shared by identities "
                        f"{prior_tag_owner!r} and {identity!r}",
                    )
                tag_owner[atom_tag] = identity
            core_key = id(core)
            prior_core_owner = core_owner.get(core_key)
            if prior_core_owner is not None and prior_core_owner != identity:
                raise CoefficientMetadataError(
                    "coefficient_parameter_alias_conflict",
                    f"one fitted coefficient parameter is shared by identities "
                    f"{prior_core_owner!r} and {identity!r}",
                )
            core_owner[core_key] = identity

            occurrence = {
                "atom_index": int(atom_index),
                "atom_tag": atom_tag,
                "parameter_path": f"leaf.{atom_index}.value",
            }
            candidate = {
                "identity": identity,
                "kind": kind,
                "name": name,
                "symbol": symbol,
                "display": display,
                "value": value,
                "dimension": dimension,
                "dimension_status": dimension_status,
                "scope": scope,
                "trainable": trainable,
                "value_source": (
                    "fitted_parameter" if trainable else "fixed_buffer"
                ),
                "dataset_id": None if dataset_id is None else str(dataset_id),
                "dataset_index": (
                    None if dataset_index is None else int(dataset_index)
                ),
                "occurrences": [occurrence],
            }
            prior = by_identity.get(identity)
            if prior is None:
                by_identity[identity] = candidate
                records.append(candidate)
                continue
            if _record_semantics(prior) != _record_semantics(candidate):
                raise CoefficientMetadataError(
                    "coefficient_identity_conflict",
                    f"coefficient identity {identity!r} has conflicting metadata",
                )
            if not math.isclose(
                float(prior["value"]), value, rel_tol=1.0e-12, abs_tol=1.0e-15
            ):
                raise CoefficientMetadataError(
                    "coefficient_value_conflict",
                    f"coefficient identity {identity!r} has conflicting values "
                    f"{prior['value']!r} and {value!r}",
                )
            prior["occurrences"].append(occurrence)

        payload = empty_coefficient_metadata(
            dimension_basis=basis,
            source=source,
            dataset_id=dataset_id,
            dataset_index=dataset_index,
        )
        payload["records"] = records
        payload["record_count"] = len(records)
        payload["symbol_count"] = len(
            {record["symbol"] for record in records if record.get("symbol")}
        )
        return normalize_coefficient_metadata(
            payload,
            require_values=True,
            units_spec=units_spec,
        )
    except CoefficientMetadataError as exc:
        return _invalid_metadata(
            exc.code,
            exc.reason,
            dimension_basis=basis,
            source=source,
            records=records,
            dataset_id=dataset_id,
            dataset_index=dataset_index,
        )
    except Exception as exc:
        return _invalid_metadata(
            "coefficient_metadata_collection_failed",
            f"coefficient metadata collection failed: {exc}",
            dimension_basis=basis,
            source=source,
            records=records,
            dataset_id=dataset_id,
            dataset_index=dataset_index,
        )


def normalize_coefficient_metadata(
    payload: Optional[Mapping[str, Any]],
    *,
    variable_names: Sequence[str] = (),
    require_values: bool = False,
    units_spec: Any = None,
) -> dict[str, Any]:
    """Validate and normalize a coefficient-metadata payload.

    ``None`` is the backward-compatible empty bundle.  Explicitly invalid or
    malformed payloads raise :class:`CoefficientMetadataError` so numerical
    consumers cannot silently evaluate a symbol with the wrong value.
    When ``units_spec`` is supplied, the stored basis, dimensions, scopes, and
    fixed values must also agree with the active units declaration.
    """

    if payload is None:
        basis = (
            tuple(
                str(value)
                for value in getattr(
                    getattr(units_spec, "unit_system", None), "base", ()
                )
            )
            if units_spec is not None
            else ()
        )
        return empty_coefficient_metadata(dimension_basis=basis)
    if not isinstance(payload, Mapping):
        raise CoefficientMetadataError(
            "coefficient_metadata_type_error", "coefficient metadata must be a mapping"
        )
    schema_raw = payload.get("schema")
    if not isinstance(schema_raw, str):
        raise CoefficientMetadataError(
            "coefficient_metadata_schema_unsupported",
            "coefficient metadata schema must be a string",
        )
    schema = schema_raw
    if schema != COEFFICIENT_METADATA_SCHEMA:
        raise CoefficientMetadataError(
            "coefficient_metadata_schema_unsupported",
            f"unsupported coefficient metadata schema {schema!r}",
        )
    if payload.get("valid") is not True:
        raise CoefficientMetadataError(
            str(payload.get("code") or "coefficient_metadata_invalid"),
            str(payload.get("reason") or "coefficient metadata is marked invalid"),
        )

    basis_raw = payload.get("dimension_basis")
    if basis_raw is None:
        basis_raw = ()
    if not isinstance(basis_raw, (list, tuple)):
        raise CoefficientMetadataError(
            "coefficient_dimension_basis_invalid",
            "coefficient dimension_basis must be a sequence",
        )
    if any(not isinstance(value, str) or not value.strip() for value in basis_raw):
        raise CoefficientMetadataError(
            "coefficient_dimension_basis_invalid",
            "coefficient dimension_basis labels must be non-empty strings",
        )
    basis = [value.strip() for value in basis_raw]
    if len(set(basis)) != len(basis):
        raise CoefficientMetadataError(
            "coefficient_dimension_basis_invalid",
            "coefficient dimension_basis contains duplicate labels",
        )
    records_raw = payload.get("records")
    if records_raw is None:
        records_raw = ()
    if not isinstance(records_raw, (list, tuple)):
        raise CoefficientMetadataError(
            "coefficient_records_invalid", "coefficient records must be a sequence"
        )

    variable_set = {str(name) for name in variable_names}
    identities: set[str] = set()
    symbols: dict[str, str] = {}
    records: list[dict[str, Any]] = []
    for index, raw in enumerate(records_raw):
        if not isinstance(raw, Mapping):
            raise CoefficientMetadataError(
                "coefficient_record_invalid",
                f"coefficient record {index} must be a mapping",
            )
        record = dict(raw)
        identity_raw = record.get("identity")
        kind_raw = record.get("kind")
        if not isinstance(identity_raw, str) or not isinstance(kind_raw, str):
            raise CoefficientMetadataError(
                "coefficient_record_invalid",
                f"coefficient record {index} identity and kind must be strings",
            )
        identity = identity_raw.strip()
        kind = kind_raw.strip().lower()
        if not identity or not kind:
            raise CoefficientMetadataError(
                "coefficient_record_invalid",
                f"coefficient record {index} requires identity and kind",
            )
        if kind not in {"free_const", "fixed_const", "scale"}:
            raise CoefficientMetadataError(
                "coefficient_kind_invalid",
                f"coefficient {identity!r} has unsupported kind {kind!r}",
            )
        if identity in identities:
            raise CoefficientMetadataError(
                "coefficient_identity_duplicate",
                f"coefficient identity {identity!r} appears more than once",
            )
        identities.add(identity)

        value = record.get("value")
        if value is None:
            if require_values:
                raise CoefficientMetadataError(
                    "coefficient_value_missing",
                    f"coefficient {identity!r} has no numeric value",
                )
            value_norm = None
        else:
            value_norm = _finite_float(value, label=f"coefficient {identity!r}")

        symbol_raw = record.get("symbol")
        symbol = None
        if symbol_raw not in (None, ""):
            if not isinstance(symbol_raw, str):
                raise CoefficientMetadataError(
                    "coefficient_symbol_invalid",
                    f"coefficient {identity!r} symbol must be a string",
                )
            symbol = validate_coefficient_symbol(symbol_raw)
            if symbol in variable_set:
                raise CoefficientMetadataError(
                    "coefficient_symbol_variable_collision",
                    f"coefficient symbol {symbol!r} collides with an input variable",
                )
            if value_norm is None and require_values:
                raise CoefficientMetadataError(
                    "coefficient_value_missing",
                    f"symbolic coefficient {symbol!r} has no numeric value",
                )
            prior_identity = symbols.get(symbol)
            if prior_identity is not None:
                raise CoefficientMetadataError(
                    "coefficient_symbol_duplicate",
                    f"coefficient symbol {symbol!r} belongs to both "
                    f"{prior_identity!r} and {identity!r}",
                )
            symbols[symbol] = identity

        dimension_raw = record.get("dimension")
        if dimension_raw is None:
            dimension = None
        else:
            try:
                dimension = _dimension_payload(dimension_raw)
            except Exception as exc:
                raise CoefficientMetadataError(
                    "coefficient_dimension_invalid",
                    f"coefficient {identity!r} has an invalid dimension: {exc}",
                ) from exc
            if len(dimension) != len(basis):
                raise CoefficientMetadataError(
                    "coefficient_dimension_rank_mismatch",
                    f"coefficient {identity!r} has dimension rank {len(dimension)}; "
                    f"expected {len(basis)}",
                )

        scope_raw = record.get("scope")
        if scope_raw in (None, ""):
            scope = None
        elif not isinstance(scope_raw, str):
            raise CoefficientMetadataError(
                "coefficient_scope_invalid",
                f"coefficient {identity!r} scope must be a string or null",
            )
        elif scope_raw.strip().lower() == "fixed":
            scope = "fixed"
        else:
            try:
                scope = normalize_free_const_scope(scope_raw, default="experiment")
            except Exception as exc:
                raise CoefficientMetadataError(
                    "coefficient_scope_invalid",
                    f"coefficient {identity!r} has invalid scope {scope_raw!r}",
                ) from exc

        display_raw = record.get("display")
        if not isinstance(display_raw, str):
            raise CoefficientMetadataError(
                "coefficient_display_invalid",
                f"coefficient {identity!r} display must be a string",
            )
        display = display_raw.strip().lower()
        if display not in {"symbol", "numeric"}:
            raise CoefficientMetadataError(
                "coefficient_display_invalid",
                f"coefficient {identity!r} has invalid display mode {display!r}",
            )
        if display == "symbol" and symbol is None:
            raise CoefficientMetadataError(
                "coefficient_symbol_missing",
                f"coefficient {identity!r} requests symbolic display without a symbol",
            )
        trainable_raw = record.get("trainable")
        if not isinstance(trainable_raw, bool):
            raise CoefficientMetadataError(
                "coefficient_trainable_invalid",
                f"coefficient {identity!r} trainable must be boolean",
            )
        trainable = trainable_raw
        name_raw = record.get("name")
        if name_raw in (None, ""):
            name = None
        elif not isinstance(name_raw, str):
            raise CoefficientMetadataError(
                "coefficient_name_invalid",
                f"coefficient {identity!r} name must be a string or null",
            )
        else:
            name = validate_coefficient_name(name_raw)
        value_source_raw = record.get("value_source")
        if not isinstance(value_source_raw, str):
            raise CoefficientMetadataError(
                "coefficient_value_source_invalid",
                f"coefficient {identity!r} value_source must be a string",
            )
        value_source = value_source_raw.strip()
        dimension_status_raw = record.get("dimension_status")
        if not isinstance(dimension_status_raw, str):
            raise CoefficientMetadataError(
                "coefficient_dimension_status_invalid",
                f"coefficient {identity!r} dimension_status must be a string",
            )
        dimension_status = dimension_status_raw.strip().lower()
        if dimension_status not in {"unavailable", "declared", "dimensionless"}:
            raise CoefficientMetadataError(
                "coefficient_dimension_status_invalid",
                f"coefficient {identity!r} has invalid dimension status "
                f"{dimension_status!r}",
            )
        if dimension is None and dimension_status != "unavailable":
            raise CoefficientMetadataError(
                "coefficient_dimension_status_invalid",
                f"coefficient {identity!r} has no dimension but status "
                f"{dimension_status!r}",
            )
        if dimension is not None and dimension_status == "unavailable":
            raise CoefficientMetadataError(
                "coefficient_dimension_status_invalid",
                f"coefficient {identity!r} has a dimension marked unavailable",
            )
        if kind in {"free_const", "fixed_const"}:
            if symbol is None or display != "symbol":
                raise CoefficientMetadataError(
                    "coefficient_named_record_invalid",
                    f"{kind} coefficient {identity!r} must retain a symbolic name",
                )
            expected_symbol = coefficient_symbol_for_name(name)
            if symbol != expected_symbol:
                raise CoefficientMetadataError(
                    "coefficient_name_invalid",
                    f"coefficient {identity!r} symbol {symbol!r} does not match "
                    f"the safe symbol {expected_symbol!r} for name {name!r}",
                )
            expected_identity = f"{kind}:{name}"
            if identity != expected_identity:
                raise CoefficientMetadataError(
                    "coefficient_identity_invalid",
                    f"coefficient identity {identity!r} does not match "
                    f"{expected_identity!r}",
                )
            if dimension is not None and dimension_status != "declared":
                raise CoefficientMetadataError(
                    "coefficient_dimension_status_invalid",
                    f"named coefficient {identity!r} with a dimension must be "
                    "marked declared",
                )
        if kind == "free_const":
            if scope not in {"experiment", "class"} or not trainable:
                raise CoefficientMetadataError(
                    "coefficient_free_record_invalid",
                    f"free coefficient {identity!r} requires experiment/class scope "
                    "and trainable=true",
                )
            if value_source != "fitted_parameter":
                raise CoefficientMetadataError(
                    "coefficient_value_source_invalid",
                    f"free coefficient {identity!r} requires fitted_parameter value source",
                )
        if kind == "fixed_const":
            if scope != "fixed" or trainable:
                raise CoefficientMetadataError(
                    "coefficient_fixed_record_invalid",
                    f"fixed coefficient {identity!r} requires fixed scope and "
                    "trainable=false",
                )
            if value_source != "fixed_buffer":
                raise CoefficientMetadataError(
                    "coefficient_value_source_invalid",
                    f"fixed coefficient {identity!r} requires fixed_buffer value source",
                )
        if kind == "scale":
            if (
                symbol is not None
                or display != "numeric"
                or not trainable
                or scope not in {"experiment", "class"}
            ):
                raise CoefficientMetadataError(
                    "coefficient_scale_record_invalid",
                    f"scale coefficient {identity!r} must be anonymous, numeric, "
                    "trainable, and experiment/class scoped",
                )
            if not identity.startswith("scale:"):
                raise CoefficientMetadataError(
                    "coefficient_identity_invalid",
                    f"scale coefficient identity {identity!r} must start with 'scale:'",
                )
            if dimension is not None and any(_fraction(value) != 0 for value in dimension):
                raise CoefficientMetadataError(
                    "coefficient_scale_dimension_invalid",
                    f"anonymous scale coefficient {identity!r} must be dimensionless",
                )
            if dimension is not None and dimension_status != "dimensionless":
                raise CoefficientMetadataError(
                    "coefficient_dimension_status_invalid",
                    f"scale coefficient {identity!r} with a dimension must be "
                    "marked dimensionless",
                )
            if value_source != "fitted_parameter":
                raise CoefficientMetadataError(
                    "coefficient_value_source_invalid",
                    f"scale coefficient {identity!r} requires fitted_parameter value source",
                )
        occurrences_raw = record.get("occurrences")
        if occurrences_raw is None:
            occurrences_raw = []
        if not isinstance(occurrences_raw, (list, tuple)) or any(
            not isinstance(item, Mapping) for item in occurrences_raw
        ):
            raise CoefficientMetadataError(
                "coefficient_occurrences_invalid",
                f"coefficient {identity!r} occurrences must be a sequence of mappings",
            )
        record.update(
            {
                "identity": identity,
                "kind": kind,
                "name": name,
                "symbol": symbol,
                "display": display,
                "value": value_norm,
                "dimension": dimension,
                "dimension_status": dimension_status,
                "scope": scope,
                "trainable": trainable,
                "value_source": value_source,
                "occurrences": [dict(item) for item in occurrences_raw],
            }
        )
        records.append(record)

    for field_name, actual in (
        ("record_count", len(records)),
        ("symbol_count", len(symbols)),
    ):
        declared = payload.get(field_name)
        if declared is None:
            continue
        declared_int = _strict_nonnegative_int(
            declared,
            code="coefficient_count_invalid",
            label=f"coefficient metadata {field_name}",
        )
        if declared_int != actual:
            raise CoefficientMetadataError(
                "coefficient_count_mismatch",
                f"coefficient metadata {field_name}={declared_int} but contains "
                f"{actual}",
            )

    dataset_index = _strict_nonnegative_int(
        payload.get("dataset_index"),
        code="coefficient_dataset_index_invalid",
        label="coefficient metadata dataset_index",
        allow_none=True,
    )
    dataset_id_raw = payload.get("dataset_id")
    if dataset_id_raw is not None and (
        not isinstance(dataset_id_raw, str) or not dataset_id_raw.strip()
    ):
        raise CoefficientMetadataError(
            "coefficient_dataset_id_invalid",
            "coefficient metadata dataset_id must be a non-empty string or null",
        )
    dataset_id = None if dataset_id_raw is None else dataset_id_raw.strip()

    for record in records:
        record_dataset_index = _strict_nonnegative_int(
            record.get("dataset_index"),
            code="coefficient_dataset_index_invalid",
            label=f"coefficient {record['identity']!r} dataset_index",
            allow_none=True,
        )
        record_dataset_id_raw = record.get("dataset_id")
        if record_dataset_id_raw is not None and (
            not isinstance(record_dataset_id_raw, str)
            or not record_dataset_id_raw.strip()
        ):
            raise CoefficientMetadataError(
                "coefficient_dataset_id_invalid",
                f"coefficient {record['identity']!r} dataset_id must be a "
                "non-empty string or null",
            )
        record_dataset_id = (
            None
            if record_dataset_id_raw is None
            else record_dataset_id_raw.strip()
        )
        if record_dataset_index != dataset_index or record_dataset_id != dataset_id:
            raise CoefficientMetadataError(
                "coefficient_dataset_identity_conflict",
                f"coefficient {record['identity']!r} dataset identity does not "
                "match its enclosing metadata bundle",
            )
        record["dataset_index"] = record_dataset_index
        record["dataset_id"] = record_dataset_id

    for field_name in ("code", "reason", "source"):
        field_value = payload.get(field_name)
        if field_value is not None and not isinstance(field_value, str):
            raise CoefficientMetadataError(
                "coefficient_metadata_field_invalid",
                f"coefficient metadata {field_name} must be a string",
            )

    normalized = {
        "schema": COEFFICIENT_METADATA_SCHEMA,
        "valid": True,
        "code": str(payload.get("code") or "coefficient_metadata_ok"),
        "reason": str(payload.get("reason") or "coefficient metadata is valid"),
        "source": str(payload.get("source") or "unspecified"),
        "dimension_basis": basis,
        "dataset_id": dataset_id,
        "dataset_index": dataset_index,
        "record_count": len(records),
        "symbol_count": len(symbols),
        "records": records,
    }
    if units_spec is not None:
        expected_basis = [
            str(value)
            for value in getattr(getattr(units_spec, "unit_system", None), "base", ())
        ]
        if basis != expected_basis:
            raise CoefficientMetadataError(
                "coefficient_dimension_basis_conflict",
                f"coefficient metadata basis {basis!r} does not match active "
                f"units basis {expected_basis!r}",
            )
        free_dims = dict(getattr(units_spec, "free_const_dims", {}) or {})
        free_scopes = dict(getattr(units_spec, "free_const_scope", {}) or {})
        fixed_dims = dict(getattr(units_spec, "fixed_const_dims", {}) or {})
        fixed_values = dict(getattr(units_spec, "fixed_const_values", {}) or {})
        for record in records:
            kind = record["kind"]
            logical_name = record.get("name")
            if kind == "scale":
                expected_dimension = ["0" for _ in expected_basis]
            elif kind == "free_const":
                if logical_name not in free_dims:
                    raise CoefficientMetadataError(
                        "coefficient_dimension_undeclared",
                        f"free coefficient {logical_name!r} has no active dimension declaration",
                    )
                expected_dimension = _dimension_payload(free_dims[logical_name])
                expected_scope = normalize_free_const_scope(
                    free_scopes.get(logical_name, "experiment"),
                    default="experiment",
                )
                if record.get("scope") != expected_scope:
                    raise CoefficientMetadataError(
                        "coefficient_scope_conflict",
                        f"free coefficient {logical_name!r} metadata scope "
                        f"{record.get('scope')!r} does not match active scope "
                        f"{expected_scope!r}",
                    )
            else:
                if logical_name not in fixed_dims:
                    raise CoefficientMetadataError(
                        "coefficient_dimension_undeclared",
                        f"fixed coefficient {logical_name!r} has no active dimension declaration",
                    )
                expected_dimension = _dimension_payload(fixed_dims[logical_name])
                if logical_name in fixed_values:
                    expected_value = _finite_float(
                        fixed_values[logical_name],
                        label=f"declared fixed coefficient {logical_name!r}",
                    )
                    if record["value"] is None:
                        raise CoefficientMetadataError(
                            "coefficient_value_missing",
                            f"fixed coefficient {logical_name!r} has no metadata value",
                        )
                    if not math.isclose(
                        float(record["value"]),
                        expected_value,
                        rel_tol=1.0e-12,
                        abs_tol=1.0e-15,
                    ):
                        raise CoefficientMetadataError(
                            "fixed_coefficient_value_conflict",
                            f"fixed coefficient {logical_name!r} metadata value "
                            f"{record['value']!r} does not match active value "
                            f"{expected_value!r}",
                        )
            if record.get("dimension") != expected_dimension:
                raise CoefficientMetadataError(
                    "coefficient_dimension_conflict",
                    f"coefficient {record['identity']!r} metadata dimension "
                    f"{record.get('dimension')!r} does not match active dimension "
                    f"{expected_dimension!r}",
                )
    return normalized


def coefficient_symbol_values(
    payload: Optional[Mapping[str, Any]],
    *,
    variable_names: Sequence[str] = (),
    units_spec: Any = None,
) -> dict[str, float]:
    """Return the named-symbol numeric substitution map from a valid bundle."""

    normalized = normalize_coefficient_metadata(
        payload,
        variable_names=variable_names,
        require_values=True,
        units_spec=units_spec,
    )
    return {
        str(record["symbol"]): float(record["value"])
        for record in normalized["records"]
        if record.get("symbol") is not None
    }


def coefficient_symbol_values_for_expression(
    payload: Optional[Mapping[str, Any]],
    expression: Any,
    *,
    variable_names: Sequence[str] = (),
    units_spec: Any = None,
) -> dict[str, float]:
    """Return exactly the named values needed by ``expression``.

    This is the relational validation boundary between a self-consistent
    metadata bundle and a particular stored expression.  Every non-input free
    symbol must have one metadata value; unused records are allowed because a
    valid simplification can eliminate a fitted coefficient.
    """

    expression_text = str(expression)
    inferred_variables = {str(name) for name in variable_names}
    inferred_variables.update(re.findall(r"\bx\d+\b", expression_text))
    values = coefficient_symbol_values(
        payload,
        variable_names=sorted(inferred_variables),
        units_spec=units_spec,
    )
    try:
        import sympy as sp

        if isinstance(expression, sp.Basic):
            parsed = expression
        else:
            normalized_expression, symbol_names = _validated_math_expression_text(
                expression_text
            )
            locals_map = {
                name: sp.Symbol(name, real=True)
                for name in sorted(inferred_variables | set(values))
            }
            function_locals = {
                "Abs": sp.Abs,
                "abs": sp.Abs,
                "acos": sp.acos,
                "acosh": sp.acosh,
                "arccos": sp.acos,
                "arccosh": sp.acosh,
                "arcsin": sp.asin,
                "arcsinh": sp.asinh,
                "arctan": sp.atan,
                "arctanh": sp.atanh,
                "asin": sp.asin,
                "asinh": sp.asinh,
                "atan": sp.atan,
                "atan2": sp.atan2,
                "atanh": sp.atanh,
                "ceiling": sp.ceiling,
                "cos": sp.cos,
                "cosh": sp.cosh,
                "erf": sp.erf,
                "exp": sp.exp,
                "floor": sp.floor,
                "ln": sp.log,
                "log": sp.log,
                "sign": sp.sign,
                "sin": sp.sin,
                "sinh": sp.sinh,
                "sqrt": sp.sqrt,
                "tan": sp.tan,
                "tanh": sp.tanh,
            }
            locals_map.update(function_locals)
            locals_map.update({"pi": sp.pi, "E": sp.E})
            for name in sorted(symbol_names):
                locals_map.setdefault(name, sp.Symbol(name, real=True))
            parsed = sp.sympify(
                normalized_expression,
                locals=locals_map,
                evaluate=False,
            )
            if not isinstance(parsed, sp.Basic):
                raise CoefficientMetadataError(
                    "coefficient_expression_parse_failed",
                    "coefficient expression did not parse to a symbolic expression",
                )
    except CoefficientMetadataError:
        raise
    except Exception as exc:
        raise CoefficientMetadataError(
            "coefficient_expression_parse_failed",
            f"could not parse expression while validating coefficient values: {exc}",
        ) from exc
    required = sorted(
        str(symbol)
        for symbol in parsed.free_symbols
        if str(symbol) not in inferred_variables
    )
    missing = [name for name in required if name not in values]
    if missing:
        raise CoefficientMetadataError(
            "coefficient_symbol_value_missing",
            "expression has no coefficient value metadata for symbols: "
            + ", ".join(missing),
        )
    return {name: values[name] for name in required}


def normalize_coefficient_metadata_by_dataset(
    payloads: Any,
    *,
    primary_payload: Optional[Mapping[str, Any]] = None,
    variable_names: Sequence[str] = (),
    units_spec: Any = None,
    expected_count: Optional[int] = None,
    expected_dataset_ids: Optional[Sequence[str]] = None,
) -> list[dict[str, Any]]:
    """Validate the topology and sharing contract of per-dataset bundles."""

    if not isinstance(payloads, (list, tuple)):
        raise CoefficientMetadataError(
            "coefficient_dataset_bundles_invalid",
            "coefficient metadata by dataset must be a sequence",
        )
    if expected_count is not None:
        count = _strict_nonnegative_int(
            expected_count,
            code="coefficient_dataset_count_invalid",
            label="expected coefficient dataset count",
        )
        if len(payloads) != count:
            raise CoefficientMetadataError(
                "coefficient_dataset_count_mismatch",
                f"coefficient metadata contains {len(payloads)} datasets; expected {count}",
            )
    expected_ids = None
    if expected_dataset_ids is not None:
        if not isinstance(expected_dataset_ids, (list, tuple)) or any(
            not isinstance(dataset_id, str) or not dataset_id.strip()
            for dataset_id in expected_dataset_ids
        ):
            raise CoefficientMetadataError(
                "coefficient_dataset_ids_invalid",
                "expected coefficient dataset_ids must be non-empty strings",
            )
        expected_ids = [dataset_id.strip() for dataset_id in expected_dataset_ids]
        if len(expected_ids) != len(payloads):
            raise CoefficientMetadataError(
                "coefficient_dataset_count_mismatch",
                f"coefficient metadata contains {len(payloads)} datasets; "
                f"expected dataset_ids has {len(expected_ids)}",
            )
        if len(set(expected_ids)) != len(expected_ids):
            raise CoefficientMetadataError(
                "coefficient_dataset_id_duplicate",
                "expected coefficient dataset_ids contains duplicates",
            )
    normalized = [
        normalize_coefficient_metadata(
            payload,
            variable_names=variable_names,
            require_values=True,
            units_spec=units_spec,
        )
        for payload in payloads
    ]
    if primary_payload is not None:
        primary = normalize_coefficient_metadata(
            primary_payload,
            variable_names=variable_names,
            require_values=True,
            units_spec=units_spec,
        )
        if not normalized or primary != normalized[0]:
            raise CoefficientMetadataError(
                "coefficient_primary_dataset_conflict",
                "top-level coefficient metadata does not match dataset bundle 0",
            )
    if not normalized:
        return normalized

    dataset_ids: set[str] = set()
    reference_basis = normalized[0]["dimension_basis"]
    reference_records = {
        record["identity"]: record for record in normalized[0]["records"]
    }

    def structure(record: Mapping[str, Any]) -> tuple[Any, ...]:
        return (
            record.get("kind"),
            record.get("name"),
            record.get("symbol"),
            record.get("display"),
            tuple(record.get("dimension") or ()),
            record.get("dimension_status"),
            record.get("scope"),
            record.get("trainable"),
            record.get("value_source"),
        )

    for dataset_index, bundle in enumerate(normalized):
        if bundle.get("dataset_index") != dataset_index:
            raise CoefficientMetadataError(
                "coefficient_dataset_index_mismatch",
                f"coefficient dataset bundle {dataset_index} declares index "
                f"{bundle.get('dataset_index')!r}",
            )
        dataset_id = bundle.get("dataset_id")
        if dataset_id is None:
            raise CoefficientMetadataError(
                "coefficient_dataset_id_missing",
                f"coefficient dataset bundle {dataset_index} has no dataset_id",
            )
        if dataset_id in dataset_ids:
            raise CoefficientMetadataError(
                "coefficient_dataset_id_duplicate",
                f"coefficient dataset_id {dataset_id!r} appears more than once",
            )
        dataset_ids.add(dataset_id)
        if expected_ids is not None and dataset_id != expected_ids[dataset_index]:
            raise CoefficientMetadataError(
                "coefficient_dataset_id_mismatch",
                f"coefficient dataset bundle {dataset_index} declares id "
                f"{dataset_id!r}; expected {expected_ids[dataset_index]!r}",
            )
        if bundle["dimension_basis"] != reference_basis:
            raise CoefficientMetadataError(
                "coefficient_dimension_basis_conflict",
                f"coefficient dataset {dataset_id!r} uses a different dimension basis",
            )
        records = {record["identity"]: record for record in bundle["records"]}
        if set(records) != set(reference_records):
            raise CoefficientMetadataError(
                "coefficient_dataset_structure_conflict",
                f"coefficient dataset {dataset_id!r} has different coefficient identities",
            )
        for identity, reference in reference_records.items():
            current = records[identity]
            if structure(current) != structure(reference):
                raise CoefficientMetadataError(
                    "coefficient_dataset_structure_conflict",
                    f"coefficient {identity!r} has conflicting per-dataset metadata",
                )
            if reference.get("scope") in {"class", "fixed"} and not math.isclose(
                float(current["value"]),
                float(reference["value"]),
                rel_tol=1.0e-12,
                abs_tol=1.0e-15,
            ):
                raise CoefficientMetadataError(
                    "coefficient_shared_value_conflict",
                    f"shared coefficient {identity!r} has conflicting dataset values",
                )
    return normalized


__all__ = [
    "COEFFICIENT_METADATA_SCHEMA",
    "CoefficientMetadataError",
    "coefficient_symbol_for_name",
    "coefficient_symbol_values",
    "coefficient_symbol_values_for_expression",
    "collect_coefficient_metadata",
    "empty_coefficient_metadata",
    "named_coefficient_symbol",
    "normalize_coefficient_metadata",
    "normalize_coefficient_metadata_by_dataset",
    "validate_coefficient_name",
    "validate_coefficient_symbol",
]
