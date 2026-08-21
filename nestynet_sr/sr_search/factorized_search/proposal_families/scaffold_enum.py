# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

from __future__ import annotations

from itertools import combinations
from typing import Any, Mapping, Sequence

from ..atom_policy import (
    coerce_atom_library,
    enrich_seed_block_from_library,
    seed_blocks_from_atom_relations,
)
from ..closures import (
    BoundClosure,
    make_bound_closure,
    make_direct_affine_closure,
    make_direct_linear_wrap_closure,
    make_direct_periodic_closure,
    make_direct_power_closure,
    make_direct_quadratic_closure,
    make_direct_rational_closure,
)
from ..expr_ast import dims_eq, is_valid_node, node_dims, node_size, node_str, simplify
from .common import dim0
from .operator_specs import OperatorSpec, family_operator_specs
from .seed_blocks import (
    SeedBlock,
    build_recursive_seed_pool,
    extend_seed_blocks_with_basis,
    make_seed_block,
    seed_anchor_blocks,
)
from .slot_binding import binding_snapshot, family_anchor_blocks, validate_bound_closure_bindings
from .types import OperatorApplication


def normalize_families(families: Sequence[str] | None) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in list(families or ()):
        token = str(raw or "").strip().lower()
        if not token or token in seen:
            continue
        seen.add(token)
        out.append(token)
    return out


def seed_anchor_nodes(
    *,
    nvars: int,
    pool_nodes: Sequence[tuple] | None,
    pool_dims: Sequence[Any] | None,
    var_dims: Sequence[Sequence[float]] | None,
    max_count: int,
) -> list[tuple[tuple, Any]]:
    return [
        (block.node, block.dim)
        for block in seed_anchor_blocks(
            nvars=nvars,
            pool_nodes=pool_nodes,
            pool_dims=pool_dims,
            var_dims=var_dims,
            max_count=max_count,
        )
    ]


def family_anchor_rows(
    family: str,
    anchor_rows: Sequence[tuple[tuple, Any]] | None,
    *,
    anchor_cap: int,
) -> list[tuple[tuple, Any]]:
    blocks = [
        SeedBlock(node=node, dim=dim, source="compat")
        for node, dim in list(anchor_rows or ())
        if isinstance(node, tuple) and node
    ]
    return [(block.node, block.dim) for block in family_anchor_blocks(family, blocks, anchor_cap=anchor_cap)]


def append_operator_application(
    out: list[OperatorApplication],
    seen: set[str],
    *,
    family: str,
    operator_id: str,
    scaffold_id: str,
    parent_node,
    hole_path: Sequence[int],
    target_mode: str,
    anchor_node=None,
    bindings: Mapping[str, Any] | None = None,
    bound_closure: BoundClosure | None = None,
    metadata: Mapping[str, Any] | None = None,
    max_scaffolds: int,
) -> None:
    if len(out) >= max(0, int(max_scaffolds)):
        return
    if not isinstance(parent_node, tuple) or not parent_node:
        return
    if not is_valid_node(parent_node):
        return
    # Include family and operator_id in dedup key so that different operator
    # families producing the same rendered parent expression both survive.
    key = f"{family}:{operator_id}:{node_str(parent_node)}"
    if key in seen:
        return
    out.append(
        OperatorApplication(
            family=str(family),
            scaffold_id=str(scaffold_id),
            parent_node=parent_node,
            hole_path=tuple(int(v) for v in tuple(hole_path or ())),
            target_mode=str(target_mode or "robust"),
            anchor_node=anchor_node if isinstance(anchor_node, tuple) else None,
            operator_id=str(operator_id),
            bindings=dict(bindings or {}),
            bound_closure=bound_closure if isinstance(bound_closure, BoundClosure) else None,
            metadata=dict(metadata or {}),
        )
    )
    seen.add(key)


def _binding_node(bindings: Mapping[str, Any] | None, key: str) -> tuple | None:
    if not isinstance(bindings, Mapping):
        return None
    value = bindings.get(str(key), None)
    if isinstance(value, SeedBlock):
        return value.node if isinstance(value.node, tuple) else None
    if isinstance(value, tuple) and value:
        return value
    return None


def _binding_nodes(bindings: Mapping[str, Any] | None, key: str) -> tuple[tuple, ...]:
    if not isinstance(bindings, Mapping):
        return ()
    value = bindings.get(str(key), None)
    if isinstance(value, SeedBlock):
        return (value.node,) if isinstance(value.node, tuple) else ()
    if isinstance(value, tuple) and value and isinstance(value[0], str):
        return (value,)
    out: list[tuple] = []
    for raw in list(value or ()):
        if isinstance(raw, SeedBlock):
            node = raw.node
        else:
            node = raw
        if isinstance(node, tuple) and node:
            out.append(node)
    return tuple(out)


def _make_linear_wrap_bound_closure(
    *,
    family: str,
    wrap_op: str,
    composition_mode: str,
    operator_token: str,
    scaffold_id: str,
    carrier_node: tuple,
    anchor_node: tuple | None,
    carrier_domain_rule: Any = None,
    anchor_role: str | None = None,
) -> BoundClosure:
    family_token = str(family or "").strip().lower()
    wrap_token = str(wrap_op or family_token).strip().lower() or family_token
    mode_token = str(composition_mode or "").strip().lower()
    kind = "base"
    if mode_token == "companion" or operator_token.endswith(":add") or operator_token.endswith("_add"):
        kind = "add"
    elif mode_token == "prefactor" or operator_token.endswith(":mul") or operator_token.endswith("_mul"):
        kind = "mul"
    return make_direct_linear_wrap_closure(
        scaffold_id=scaffold_id,
        family=family_token,
        wrap_kind=kind,
        wrap_op=wrap_token,
        hole_node=carrier_node or ("const", 1.0),
        feature_node=(wrap_token, carrier_node or ("const", 1.0)),
        anchor_node=anchor_node,
        carrier_domain_rule=carrier_domain_rule,
        anchor_role=anchor_role,
    )


def build_operator_bound_closure(
    *,
    spec: OperatorSpec | None = None,
    family: str,
    operator_id: str,
    scaffold_id: str,
    parent_node: tuple,
    anchor_node: tuple | None,
    bindings: Mapping[str, Any] | None,
    metadata: Mapping[str, Any] | None,
) -> BoundClosure:
    family_token = str(family or "").strip().lower()
    operator_token = str(operator_id or "").strip().lower()
    meta = dict(metadata or {})
    spec_token = spec if isinstance(spec, OperatorSpec) else None
    operator_kind = str(getattr(spec_token, "operator_kind", meta.get("operator_kind", "")) or "").strip().lower()
    composition_mode = str(getattr(spec_token, "composition_mode", meta.get("composition_mode", "")) or "").strip().lower()
    wrap_op = str(getattr(spec_token, "wrap_op", meta.get("wrap_op", "")) or "").strip().lower()
    exponent = getattr(spec_token, "exponent", meta.get("exponent", None))
    carrier_domain_rule = getattr(spec_token, "carrier_domain_rule", None)
    anchor_role = getattr(spec_token, "anchor_role", None)
    if not operator_kind:
        form = str(meta.get("form", "") or "").strip().lower()
        if family_token == "periodic":
            operator_kind = "harmonic_wrap"
        elif family_token in {"exp", "log"}:
            operator_kind = "anchored_unary_wrap" if form.endswith(("_add", "_mul")) else "unary_wrap"
            wrap_op = wrap_op or family_token
            if not composition_mode:
                if form.endswith("_add"):
                    composition_mode = "companion"
                elif form.endswith("_mul"):
                    composition_mode = "prefactor"
                else:
                    composition_mode = "base"
            if carrier_domain_rule is None and family_token == "log":
                carrier_domain_rule = "positive_output"
            if anchor_role is None and composition_mode == "companion":
                anchor_role = "companion"
        elif family_token == "rational":
            operator_kind = "fractional_head"
        elif family_token == "power":
            operator_kind = "power_wrap"
        elif family_token == "quadratic":
            operator_kind = "quadratic_wrap"
        elif family_token == "affine":
            operator_kind = "affine_latent"
    carrier_node = _binding_node(bindings, "carrier") or ("const", 1.0)
    if operator_kind == "harmonic_wrap":
        periodic_kind = wrap_op or ("sin" if ":sin" in operator_token else "cos")
        envelope_node = _binding_node(bindings, "envelope")
        companion_nodes = _binding_nodes(bindings, "companions")
        if composition_mode == "companion" and isinstance(anchor_node, tuple):
            companion_nodes = (*companion_nodes, anchor_node)
        if composition_mode == "prefactor" and isinstance(anchor_node, tuple) and envelope_node is None:
            envelope_node = anchor_node
        return make_direct_periodic_closure(
            scaffold_id=scaffold_id,
            periodic_kind=periodic_kind,
            hole_node=carrier_node or ("const", 1.0),
            feature_node=(periodic_kind, carrier_node or ("const", 1.0)),
            anchor_node=anchor_node,
            envelope_node=envelope_node,
            companion_nodes=companion_nodes,
            harmonic_feature_nodes=(
                ("cos", carrier_node or ("const", 1.0)),
                ("sin", carrier_node or ("const", 1.0)),
            ),
            expr=None,
        )
    if operator_kind in {"unary_wrap", "anchored_unary_wrap"}:
        return _make_linear_wrap_bound_closure(
            family=family_token,
            wrap_op=wrap_op or family_token,
            composition_mode=composition_mode,
            operator_token=operator_token,
            scaffold_id=scaffold_id,
            carrier_node=carrier_node or ("const", 1.0),
            anchor_node=anchor_node,
            carrier_domain_rule=carrier_domain_rule,
            anchor_role=anchor_role,
        )
    if family_token == "affine":
        return make_direct_affine_closure(
            scaffold_id=scaffold_id,
            term_nodes=_binding_nodes(bindings, "terms"),
        )
    if operator_kind == "fractional_head":
        if composition_mode in {"fractional", "base"} and operator_token == "rational:affine":
            return make_bound_closure(
                closure_id="rational:affine_fractional_head",
                family="rational",
                head_solver="fractional_linear",
                slot_specs=(),
                bindings={},
                diagnostics={"scaffold_id": str(scaffold_id)},
                metadata={"scaffold_id": str(scaffold_id), "form": "rational_affine"},
            )
        numerator = _binding_node(bindings, "numerator") or ("const", 0.0)
        denominator = _binding_node(bindings, "denominator") or ("const", 0.0)
        if composition_mode == "denominator_companion" and denominator == ("const", 0.0) and isinstance(anchor_node, tuple):
            denominator = anchor_node
        if composition_mode == "numerator_companion" and numerator == ("const", 0.0) and isinstance(anchor_node, tuple):
            numerator = anchor_node
        return make_direct_rational_closure(
            scaffold_id=scaffold_id,
            u_node=numerator,
            v_node=denominator,
        )
    if operator_kind == "quadratic_wrap":
        quadratic_kind = "sqrt_mul" if composition_mode == "prefactor" or operator_token.endswith(":sqrt_mul") else "sqrt"
        base_nodes = _binding_nodes(bindings, "bases")
        if not base_nodes and carrier_node is not None:
            base_nodes = (carrier_node,)
        return make_direct_quadratic_closure(
            scaffold_id=scaffold_id,
            quadratic_kind=quadratic_kind,
            base_nodes=base_nodes or (("const", 1.0),),
            anchor_node=anchor_node,
        )
    if operator_kind == "power_wrap":
        power_kind = wrap_op or (operator_token.split(":", 1)[1] if ":" in operator_token else operator_token)
        if composition_mode == "prefactor" and not power_kind.endswith("_mul"):
            power_kind = f"{power_kind}_mul"
        exponent_map = {
            "sqrt": 0.5,
            "sqrt_mul": 0.5,
            "invsqrt": -0.5,
            "invsqrt_mul": -0.5,
            "inv": -1.0,
            "inv_mul": -1.0,
            "neg2": -2.0,
            "neg2_mul": -2.0,
            "sqr": 2.0,
            "sqr_mul": 2.0,
        }
        return make_direct_power_closure(
            scaffold_id=scaffold_id,
            power_kind=power_kind,
            exponent=float(exponent if exponent is not None else exponent_map.get(power_kind, 0.5)),
            hole_node=carrier_node or ("const", 1.0),
            anchor_node=anchor_node,
        )
    return make_bound_closure(
        closure_id=f"{family_token or 'unknown'}:{operator_token or 'literal'}",
        family=family_token or "unknown",
        head_solver="unknown",
        slot_specs=(),
        bindings={
            "expr": parent_node,
            "anchor": anchor_node,
        },
        metadata={
            "operator_id": str(operator_id),
            "scaffold_id": str(scaffold_id),
            "form": str(meta.get("form", "") or ""),
        },
    )


def _render_operator_template(template: Any, *, carrier_node: tuple | None, anchor_node: tuple | None) -> Any:
    if template == "__CARRIER__":
        return carrier_node if isinstance(carrier_node, tuple) else ("const", 1.0)
    if template == "__ANCHOR__":
        return anchor_node if isinstance(anchor_node, tuple) else ("const", 1.0)
    if isinstance(template, tuple):
        return tuple(
            _render_operator_template(child, carrier_node=carrier_node, anchor_node=anchor_node)
            for child in template
        )
    return template


def _render_affine_parent_node(term_nodes: Sequence[tuple] | None) -> tuple:
    nodes = [node for node in list(term_nodes or ()) if isinstance(node, tuple) and is_valid_node(node)]
    if not nodes:
        return ("const", 0.0)
    cur = nodes[0]
    for node in nodes[1:]:
        cur = simplify(("add", cur, node))
    return cur


def _anchor_allowed_for_spec(spec: OperatorSpec, *, anchor_dim: Any, dim0_value: Any) -> bool:
    rule = str(spec.anchor_dim_rule or "").strip().lower()
    if not rule:
        return True
    if rule == "dim0":
        if dim0_value is None:
            return True
        return anchor_dim is not None and dims_eq(anchor_dim, dim0_value)
    return True


def _quadratic_base_bindings(block: SeedBlock | None) -> tuple[SeedBlock, ...]:
    if not isinstance(block, SeedBlock):
        return ()
    rows = list(dict(block.metadata or {}).get("bases", []) or [])
    out: list[SeedBlock] = []
    for raw in rows:
        if not isinstance(raw, Mapping):
            continue
        node = raw.get("node", None)
        if not isinstance(node, tuple):
            expr = raw.get("expr", None)
            if isinstance(expr, str):
                continue
            continue
        out.append(
            SeedBlock(
                node=node,
                dim=raw.get("dim", None),
                source=str(raw.get("source", "quadratic_base")),
                builder=str(raw.get("builder", "identity")),
                active_vars=tuple(int(v) for v in tuple(raw.get("active_vars", ()) or ())),
                domain_tags=tuple(str(v) for v in tuple(raw.get("domain_tags", ()) or ())),
                metadata=dict(raw.get("metadata", {}) or {}),
            )
        )
    if out:
        return tuple(out)
    base_nodes = [node for node in list(dict(block.metadata or {}).get("base_nodes", []) or ()) if isinstance(node, tuple)]
    if base_nodes:
        base_dims = list(dict(block.metadata or {}).get("base_dims", []) or [])
        return tuple(
            SeedBlock(
                node=node,
                dim=base_dims[idx] if idx < len(base_dims) else None,
                source="quadratic_base",
                builder="identity",
            )
            for idx, node in enumerate(base_nodes)
        )
    return ()


def _seed_source_rank(block: SeedBlock) -> int:
    source = str(getattr(block, "source", "") or "")
    origin = str(dict(getattr(block, "metadata", {}) or {}).get("origin", "") or "")
    token = origin or source
    if token.startswith("aux:"):
        return 1
    if token.startswith("basis:active"):
        return 3
    if token.startswith("basis:beam"):
        return 4
    return 0


def _anchor_sort_key(block: SeedBlock) -> tuple[Any, ...]:
    node = block.node
    node_key = str(node_str(node))
    source_rank = _seed_source_rank(block)
    is_const = 1 if node == ("const", 1.0) else 0
    builder = str(block.builder or "")
    builder_rank = {
        "identity": 0,
        "basis_head": 1,
        "basis_atom": 1,
        "product": 2,
        "basis_latent": 2,
        "monomial": 2,
        "quadratic": 3,
    }.get(builder, 4)
    try:
        nonlinear_depth = int(dict(block.metadata or {}).get("nonlinear_depth", 0) or 0)
    except Exception:
        nonlinear_depth = 0
    try:
        size = max(1, int(node_size(node)))
    except Exception:
        size = 99
    active_vars = -int(len(tuple(block.active_vars or ())))
    policy_rank = -_policy_block_score(block, "anchor", "envelope", "prefactor", "affine.terms")
    return (policy_rank, is_const, builder_rank, nonlinear_depth, source_rank, size, active_vars, node_key)


def _policy_block_score(block: SeedBlock, *slot_keys: str) -> float:
    if not isinstance(block, SeedBlock):
        return 0.0
    meta = dict(block.metadata or {})
    score = 0.0
    try:
        score = max(score, float(meta.get("policy_score", 0.0) or 0.0))
    except Exception:
        pass
    for key_name in ("policy_slot_scores", "slot_scores"):
        raw = meta.get(key_name, {})
        if not isinstance(raw, Mapping):
            continue
        for slot in tuple(slot_keys or ()):
            try:
                score = max(score, float(dict(raw).get(str(slot), 0.0) or 0.0))
            except Exception:
                pass
    policy_record = meta.get("atom_policy_record", None)
    if isinstance(policy_record, Mapping):
        for key_name in ("slot_scores", "role_scores"):
            raw = dict(policy_record).get(key_name, {})
            if not isinstance(raw, Mapping):
                continue
            for slot in tuple(slot_keys or ()):
                try:
                    score = max(score, float(dict(raw).get(str(slot), 0.0) or 0.0))
                except Exception:
                    pass
    emergent_atom = meta.get("emergent_atom", None)
    if isinstance(emergent_atom, Mapping):
        kind = str(dict(emergent_atom).get("kind", "") or "")
        if kind == "target_term" and any(str(slot).startswith("affine") or str(slot) in {"anchor", "prefactor"} for slot in slot_keys):
            score = max(score, 0.5)
        if kind == "carrier" and any(str(slot) in {"anchor", "envelope", "prefactor"} for slot in slot_keys):
            score = max(score, 0.6)
        if kind == "dimensionless_feature" and any("carrier" in str(slot) or "modulator" in str(slot) for slot in slot_keys):
            score = max(score, 0.45)
    return float(score)


def _is_ratio_like_node(node: Any) -> bool:
    if not (isinstance(node, tuple) and node and str(node[0]) == "div" and len(node) >= 3):
        return False
    left, right = node[1], node[2]
    if not (isinstance(left, tuple) and left and isinstance(right, tuple) and right):
        return False
    if str(left[0]) == "const" or str(right[0]) == "const":
        return False
    return True


def _is_ratio_square_node(node: Any) -> bool:
    return bool(
        isinstance(node, tuple)
        and node
        and str(node[0]) == "sqr"
        and len(node) >= 2
        and _is_ratio_like_node(node[1])
    )


def _is_wrapped_node(node: Any) -> bool:
    return bool(isinstance(node, tuple) and node and str(node[0]) in {"exp", "sqr", "sqrt", "sin", "cos", "log"})


def _is_trig_root_node(node: Any) -> bool:
    return bool(isinstance(node, tuple) and node and str(node[0]) in {"sin", "cos"})


def _carrier_sort_key(spec: OperatorSpec, block: SeedBlock) -> tuple[Any, ...]:
    node = block.node
    node_key = str(node_str(node))
    source_rank = _seed_source_rank(block)
    is_const = 1 if node == ("const", 1.0) else 0
    builder = str(block.builder or "")
    builder_rank = {
        "identity": 0,
        "basis_head": 1,
        "basis_atom": 1,
        "product": 2,
        "basis_latent": 2,
        "monomial": 3,
        "quadratic": 4,
    }.get(builder, 4)
    nonlinear_depth = 0
    try:
        nonlinear_depth = int(dict(block.metadata or {}).get("nonlinear_depth", 0) or 0)
    except Exception:
        nonlinear_depth = 0
    try:
        size = max(1, int(node_size(node)))
    except Exception:
        size = 99
    active_vars = -int(len(tuple(block.active_vars or ())))
    builder_depth = max(0, int(dict(block.metadata or {}).get("builder_depth", 0) or 0))
    domain_tags = {str(v) for v in tuple(getattr(block, "domain_tags", ()) or ())}
    slot_tokens = (
        f"{str(spec.family or '')}.{str(spec.carrier_slot or 'carrier')}",
        f"{str(spec.operator_id or '')}.{str(spec.carrier_slot or 'carrier')}",
        str(spec.carrier_slot or "carrier"),
        str(spec.carrier_role or "carrier"),
        "completed_modulator",
    )
    policy_rank = -_policy_block_score(block, *slot_tokens)
    if str(spec.operator_kind or "") == "quadratic_wrap":
        is_quadratic_builder = 0 if builder == "quadratic" else 1
        meta = dict(block.metadata or {})
        base_nodes = tuple(meta.get("base_nodes", ()) or ())
        base_affine_kinds = tuple(meta.get("base_affine_kinds", ()) or ())
        base_arity = len(base_nodes)
        difference_supports: list[set[int]] = []
        for kind, base_node in zip(base_affine_kinds, base_nodes):
            if str(kind or "") != "difference":
                continue
            if not (
                isinstance(base_node, tuple)
                and len(base_node) >= 3
                and str(base_node[0]) == "sub"
                and isinstance(base_node[1], tuple)
                and isinstance(base_node[2], tuple)
                and str(base_node[1][0]) == "var"
                and str(base_node[2][0]) == "var"
            ):
                continue
            difference_supports.append({int(base_node[1][1]), int(base_node[2][1])})
        difference_base_count = len(difference_supports)
        difference_group = 0 if difference_base_count > 0 else 1
        difference_overlap = (
            sum(
                len(left.intersection(right))
                for idx, left in enumerate(difference_supports)
                for right in difference_supports[idx + 1 :]
            )
            if difference_supports
            else 99
        )
        disjoint_difference_rank = 0 if difference_base_count >= 2 and difference_overlap == 0 else 1
        difference_size = int(size) if difference_base_count > 0 else 0
        difference_active_vars = int(-active_vars) if difference_base_count > 0 else 0
        return (
            policy_rank,
            is_quadratic_builder,
            source_rank,
            is_const,
            disjoint_difference_rank,
            difference_overlap,
            difference_group,
            difference_size,
            difference_active_vars,
            -int(difference_base_count),
            builder_depth,
            nonlinear_depth,
            -int(base_arity),
            active_vars,
            size,
            node_key,
        )
    if str(spec.operator_kind or "") == "power_wrap":
        try:
            exponent = float(getattr(spec, "exponent", 0.0) or 0.0)
        except Exception:
            exponent = 0.0
        if exponent in {-0.5, -1.0, -2.0}:
            if "positive_output" in domain_tags:
                domain_rank = 0
            elif "nonnegative_output" in domain_tags:
                domain_rank = 1
            else:
                domain_rank = 2
            ratio_square_rank = 0 if _is_ratio_square_node(node) else 1
            ratio_rank = 0 if _is_ratio_like_node(node) else 1
        elif exponent == 0.5:
            domain_rank = 0 if ("nonnegative_output" in domain_tags or "positive_output" in domain_tags) else 1
            ratio_square_rank = 1
            ratio_rank = 1
        else:
            domain_rank = 0
            ratio_square_rank = 1
            ratio_rank = 1
        recursive_rank = 0 if builder in {"basis_head", "basis_latent", "quadratic", "monomial", "product"} else 1
        return (
            policy_rank,
            is_const,
            ratio_square_rank,
            ratio_rank,
            domain_rank,
            recursive_rank,
            source_rank,
            builder_depth,
            nonlinear_depth,
            builder_rank,
            active_vars,
            size,
            node_key,
        )
    if str(spec.operator_kind or "") in {"harmonic_wrap", "unary_wrap", "anchored_unary_wrap"}:
        return (policy_rank, is_const, nonlinear_depth, builder_rank, source_rank, size, active_vars, node_key)
    return (policy_rank, is_const, builder_rank, nonlinear_depth, source_rank, size, active_vars, node_key)


def _is_recursive_wrap_carrier(block: SeedBlock) -> bool:
    if not isinstance(block, SeedBlock):
        return False
    builder = str(block.builder or "")
    meta = dict(block.metadata or {})
    builder_depth = max(0, int(meta.get("builder_depth", 0) or 0))
    nonlinear_depth = max(0, int(meta.get("nonlinear_depth", 0) or 0))
    product_arity = max(0, int(meta.get("product_arity", 0) or 0))
    try:
        size = max(1, int(node_size(block.node)))
    except Exception:
        size = 1
    if builder in {"product", "monomial", "quadratic", "affine", "basis_latent"}:
        return True
    if builder in {"basis_head", "basis_atom"} and (size > 1 or nonlinear_depth > 0 or builder_depth > 0 or product_arity > 1):
        return True
    return bool(builder_depth > 0 or nonlinear_depth > 0 or product_arity > 1)


def _shortlist_wrap_carriers(rows: Sequence[SeedBlock], *, limit: int) -> list[SeedBlock]:
    rows_sorted = list(rows or ())
    limit_i = max(0, int(limit))
    if limit_i <= 0 or len(rows_sorted) <= limit_i:
        return rows_sorted[:limit_i] if limit_i > 0 else []
    if limit_i == 1:
        return rows_sorted[:1]

    recursive_rows = [block for block in rows_sorted if _is_recursive_wrap_carrier(block)]
    simple_rows = [block for block in rows_sorted if not _is_recursive_wrap_carrier(block)]

    out: list[SeedBlock] = []
    seen: set[str] = set()

    def _add(block: SeedBlock | None) -> None:
        if not isinstance(block, SeedBlock):
            return
        key = str(node_str(block.node))
        if key in seen:
            return
        seen.add(key)
        out.append(block)

    if simple_rows:
        _add(simple_rows[0])
    if recursive_rows:
        _add(recursive_rows[0])
    lead_simple_vars = {int(v) for v in tuple(simple_rows[0].active_vars or ())} if simple_rows else set()
    if recursive_rows and lead_simple_vars:
        for block in recursive_rows[1:]:
            block_vars = {int(v) for v in tuple(block.active_vars or ())}
            if block_vars and block_vars.isdisjoint(lead_simple_vars):
                if len(out) >= limit_i:
                    break
                _add(block)
                break

    for block in rows_sorted:
        if len(out) >= limit_i:
            break
        _add(block)

    return out[:limit_i]


def _prefactor_harmonic_anchor_sort_key(block: SeedBlock) -> tuple[Any, ...]:
    node = block.node
    op = str(node[0]) if isinstance(node, tuple) and node else ""
    try:
        size = max(1, int(node_size(node)))
    except Exception:
        size = 99
    active_vars = -int(len(tuple(block.active_vars or ())))
    if op == "var":
        kind_rank = 0
    elif op == "mul":
        kind_rank = 1
    elif node == ("const", 1.0):
        kind_rank = 2
    else:
        kind_rank = 3
    policy_rank = -_policy_block_score(block, "periodic.envelope", "periodic.anchor", "envelope", "prefactor")
    return (policy_rank, kind_rank, size, active_vars, str(node_str(node)))


def _prefactor_harmonic_carrier_sort_key(anchor_block: SeedBlock, block: SeedBlock) -> tuple[Any, ...]:
    anchor_vars = {int(v) for v in tuple(anchor_block.active_vars or ())}
    carrier_vars = {int(v) for v in tuple(block.active_vars or ())}
    overlap_rank = 1 if anchor_vars.intersection(carrier_vars) else 0
    is_const = 1 if block.node == ("const", 1.0) else 0
    is_recursive = 0 if _is_recursive_wrap_carrier(block) else 1
    op = str(block.node[0]) if isinstance(block.node, tuple) and block.node else ""
    is_raw_var = 1 if op == "var" else 0
    try:
        size = max(1, int(node_size(block.node)))
    except Exception:
        size = 99
    active_vars = -int(len(tuple(block.active_vars or ())))
    policy_rank = -_policy_block_score(
        block,
        "periodic.carrier",
        "periodic:sin_mul.carrier",
        "periodic:cos_mul.carrier",
        "carrier_argument",
    )
    return (
        policy_rank,
        is_const,
        is_recursive,
        overlap_rank,
        is_raw_var,
        size,
        active_vars,
        str(node_str(block.node)),
    )


def _basis_seed_mode_token(raw: Any) -> str:
    token = str(raw or "merged").strip().lower()
    if token in {"core", "core_only", "canonical", "typed_core", "none"}:
        return "core_only"
    if token in {"basis", "basis_augmented", "augmented", "basis_only"}:
        return "basis_augmented"
    return "merged"


def _is_aux_seed_block(block: SeedBlock) -> bool:
    source = str(getattr(block, "source", "") or "")
    origin = str(dict(getattr(block, "metadata", {}) or {}).get("origin", "") or "")
    return source.startswith("aux:") or origin.startswith("aux:")


def _merge_seed_blocks_unique(*rows: Sequence[SeedBlock], limit: int | None = None) -> list[SeedBlock]:
    out: list[SeedBlock] = []
    seen: set[str] = set()
    index_by_key: dict[str, int] = {}
    for group in rows:
        for block in list(group or ()):
            if not isinstance(block, SeedBlock):
                continue
            key = str(node_str(block.node))
            if key in seen:
                idx = index_by_key.get(key)
                if idx is not None and 0 <= idx < len(out) and _is_aux_seed_block(block) and not _is_aux_seed_block(out[idx]):
                    out[idx] = block
                continue
            seen.add(key)
            index_by_key[key] = len(out)
            out.append(block)
            if limit is not None and len(out) >= int(limit):
                return out
    return out


def _sanitize_aux_seed_blocks(
    aux_seed_blocks: Sequence[Any] | None,
    *,
    var_dims,
    limit: int,
    atom_library: Any = None,
) -> list[SeedBlock]:
    out: list[SeedBlock] = []
    seen: set[str] = set()
    for raw in list(aux_seed_blocks or ()):
        node = None
        dim = None
        source = "aux:emergent"
        builder = "identity"
        metadata: dict[str, Any] = {"origin": "aux:emergent"}
        if isinstance(raw, SeedBlock):
            node = raw.node
            dim = raw.dim
            source = str(raw.source or "aux:emergent")
            builder = str(raw.builder or "identity")
            metadata.update(dict(raw.metadata or {}))
        elif isinstance(raw, Mapping):
            node = raw.get("node", None)
            dim = raw.get("dim", None)
            source = str(raw.get("source", "aux:emergent") or "aux:emergent")
            builder = str(raw.get("builder", "identity") or "identity")
            metadata.update(dict(raw.get("metadata", {}) or {}))
        elif isinstance(raw, tuple):
            node = raw
        if not isinstance(node, tuple) or not node or not is_valid_node(node):
            continue
        simp = simplify(node)
        if not isinstance(simp, tuple) or not is_valid_node(simp):
            continue
        key = str(node_str(simp))
        if key in seen:
            continue
        if dim is None and var_dims is not None:
            try:
                dim = node_dims(simp, var_dims)
            except Exception:
                dim = None
        metadata.setdefault("origin", "aux:emergent")
        out.append(
            enrich_seed_block_from_library(
                make_seed_block(
                    simp,
                    dim=dim,
                    source=source if source.startswith("aux:") else f"aux:emergent:{source}",
                    builder=builder,
                    metadata=metadata,
                ),
                coerce_atom_library(atom_library),
            )
        )
        seen.add(key)
        if len(out) >= max(0, int(limit)):
            break
    return out


def _carrier_candidates_for_spec(
    spec: OperatorSpec,
    *,
    target_dim: Any,
    anchor_dim: Any,
    dim0_value: Any,
    seed_blocks: Sequence[SeedBlock],
    var_dims,
    limit: int,
) -> list[SeedBlock]:
    if not callable(spec.carrier_dim_resolver):
        return []
    desired_dim = spec.carrier_dim_resolver(target_dim, anchor_dim, dim0_value)
    rows: list[SeedBlock] = []
    for block in list(seed_blocks or ()):
        if desired_dim is None:
            rows.append(block)
            continue
        if block.dim is None:
            if var_dims is not None:
                try:
                    block_dim = node_dims(block.node, var_dims)
                except Exception:
                    block_dim = None
            else:
                block_dim = None
        else:
            block_dim = block.dim
        if block_dim is None or not dims_eq(block_dim, desired_dim):
            continue
        rows.append(block)
    if not rows:
        return []
    if any(block.node != ("const", 1.0) for block in rows):
        rows = [block for block in rows if block.node != ("const", 1.0)]
    if str(spec.operator_kind or "") == "quadratic_wrap":
        quad_rows = [block for block in rows if str(block.builder or "") == "quadratic"]
        if quad_rows:
            rows = quad_rows
    if str(spec.operator_kind or "") == "harmonic_wrap":
        plain_rows = [block for block in rows if not _is_trig_root_node(block.node)]
        if plain_rows:
            rows = plain_rows
    rows.sort(key=lambda block: _carrier_sort_key(spec, block))
    limit_i = max(1, int(limit))
    if str(spec.operator_kind or "") in {"harmonic_wrap", "unary_wrap", "anchored_unary_wrap"}:
        return _shortlist_wrap_carriers(rows, limit=limit_i)
    return rows[:limit_i]


def _recursive_seed_pool_kwargs(
    families: Sequence[str] | None,
    *,
    anchors_per_family: int,
) -> dict[str, Any]:
    family_set = {str(token or "").strip().lower() for token in list(families or ()) if str(token or "").strip()}
    deep_recursive = bool(family_set.intersection({"periodic", "power", "quadratic", "affine"}))
    quadratic_recursive = bool(family_set.intersection({"power", "quadratic", "affine"}))
    affine_recursive = bool(family_set.intersection({"affine", "power", "quadratic"}))
    return {
        "rounds": 2 if deep_recursive else 1,
        "include_product": True,
        "include_monomial": True,
        "include_quadratic": quadratic_recursive,
        "include_affine": affine_recursive,
        "product_max_arity": 3 if deep_recursive else 2,
        "product_limit": max(6, int(anchors_per_family) * 3),
        "monomial_limit": max(8, int(anchors_per_family) * 4),
        "quadratic_max_arity": 3,
        "quadratic_limit": max(6, int(anchors_per_family) * 3),
        "affine_max_arity": 3,
        "affine_limit": max(6, int(anchors_per_family) * 3),
        "max_builder_depth": 3 if deep_recursive else 2,
        "max_nonlinear_depth": 2 if quadratic_recursive else 1,
    }


def _append_unique_dim(out: list[Any], dim: Any) -> None:
    if dim is None:
        return
    for existing in out:
        if dims_eq(existing, dim):
            return
    out.append(dim)


def _quadratic_required_expr_dims(
    families: Sequence[str],
    *,
    target_dim: Any,
    dim0_value: Any,
    anchor_blocks: Sequence[SeedBlock],
) -> list[Any]:
    out: list[Any] = []
    for family in list(families or ()):
        for spec in family_operator_specs(family):
            resolver = getattr(spec, "carrier_dim_resolver", None)
            if not callable(resolver):
                continue
            anchor_mode = str(getattr(spec, "anchor_mode", "") or "none").strip().lower()
            if anchor_mode == "per_anchor":
                for anchor_block in list(anchor_blocks or ()):
                    anchor_dim = getattr(anchor_block, "dim", None)
                    if not _anchor_allowed_for_spec(spec, anchor_dim=anchor_dim, dim0_value=dim0_value):
                        continue
                    _append_unique_dim(out, resolver(target_dim, anchor_dim, dim0_value))
            else:
                _append_unique_dim(out, resolver(target_dim, None, dim0_value))
    return out


def _affine_term_combos(
    spec: OperatorSpec,
    *,
    target_dim: Any,
    dim0_value: Any,
    seed_blocks: Sequence[SeedBlock],
    var_dims,
    limit: int,
) -> list[tuple[SeedBlock, ...]]:
    out: list[tuple[SeedBlock, ...]] = []
    seen: set[str] = set()
    max_arity = max(1, int(getattr(spec, "subset_max_arity", 1) or 1))

    def _combo_key(combo: Sequence[SeedBlock]) -> str:
        return "|".join(str(node_str(term.node)) for term in combo)

    def _append_combo(combo: Sequence[SeedBlock]) -> bool:
        if len(out) >= int(limit):
            return False
        combo_tuple = tuple(block for block in tuple(combo or ()) if isinstance(block, SeedBlock))
        if not combo_tuple:
            return False
        combo_key = _combo_key(combo_tuple)
        if combo_key in seen:
            return False
        seen.add(combo_key)
        out.append(combo_tuple)
        return True

    aux_term_rows: list[SeedBlock] = []
    for block in list(seed_blocks or ()):
        if not isinstance(block, SeedBlock) or not _is_aux_seed_block(block) or block.node == ("const", 1.0):
            continue
        block_dim = block.dim
        if block_dim is None and var_dims is not None:
            try:
                block_dim = node_dims(block.node, var_dims)
            except Exception:
                block_dim = None
        if target_dim is not None and (block_dim is None or not dims_eq(block_dim, target_dim)):
            continue
        aux_term_rows.append(block)
    def _aux_source_text(block: SeedBlock) -> str:
        parts = [str(block.source or ""), str(dict(block.metadata or {}).get("origin", "") or "")]
        for child in list(dict(block.metadata or {}).get("children", []) or ()):
            if isinstance(child, Mapping):
                parts.append(str(child.get("source", "") or ""))
                child_meta = child.get("metadata", {})
                if isinstance(child_meta, Mapping):
                    parts.append(str(child_meta.get("origin", "") or ""))
        return " ".join(parts)

    def _aux_direct_rank(block: SeedBlock) -> tuple[Any, ...]:
        text = _aux_source_text(block)
        if "target_term" in text:
            kind_rank = 0
        elif "carrier" in text:
            kind_rank = 1
        else:
            kind_rank = 2
        return (kind_rank, *_carrier_sort_key(spec, block))

    def _aux_product_rank(block: SeedBlock) -> tuple[Any, ...]:
        text = _aux_source_text(block)
        meta = dict(block.metadata or {})
        has_carrier = "carrier" in text
        has_target = "target_term" in text
        has_dimless = "dimensionless_feature" in text
        if "aux:policy" in text or isinstance(meta.get("atom_policy_relation", None), Mapping):
            kind_rank = -1
        elif has_carrier and has_dimless:
            kind_rank = 0
        elif has_target and has_dimless:
            kind_rank = 1
        elif has_carrier:
            kind_rank = 2
        else:
            kind_rank = 3
        return (kind_rank, *_carrier_sort_key(spec, block))

    aux_term_rows.sort(key=lambda block: _carrier_sort_key(spec, block))
    protected_aux_limit = min(int(limit), max(0, min(12, len(aux_term_rows) + 2)))
    direct_aux_terms = sorted(
        [block for block in aux_term_rows if str(block.builder or "") == "identity"],
        key=_aux_direct_rank,
    )
    product_aux_terms = sorted(
        [block for block in aux_term_rows if str(block.builder or "") == "product"],
        key=_aux_product_rank,
    )
    policy_product_terms = [
        block
        for block in product_aux_terms
        if isinstance(dict(block.metadata or {}).get("atom_policy_relation", None), Mapping)
    ]
    policy_product_keys = {str(node_str(block.node)) for block in policy_product_terms}
    generic_product_terms = [
        block
        for block in product_aux_terms
        if str(node_str(block.node)) not in policy_product_keys
    ]
    if policy_product_terms:
        # Policy relation atoms are already target-dimension candidates, but
        # the useful FSS move is often a small affine span: a retained target
        # term plus a relation term.  Keep room for those two-term probes
        # before spending the whole protected budget on relation singletons.
        direct_beam = 1 if protected_aux_limit <= 6 else 2
        relation_beam = 1 if protected_aux_limit <= 6 else 2
        combo_beam = min(len(policy_product_terms), max(4, min(8, protected_aux_limit)))
        for block in direct_aux_terms[:direct_beam]:
            if len(out) >= protected_aux_limit:
                break
            _append_combo((block,))
        for block in policy_product_terms[:relation_beam]:
            if len(out) >= protected_aux_limit:
                break
            _append_combo((block,))
        combo_products = policy_product_terms[:combo_beam]
        tail_products = policy_product_terms[relation_beam:12]
    else:
        for block in direct_aux_terms[:2]:
            if len(out) >= protected_aux_limit:
                break
            _append_combo((block,))
        for block in generic_product_terms[:1]:
            if len(out) >= protected_aux_limit:
                break
            _append_combo((block,))
        combo_products = generic_product_terms[:4]
        tail_products = generic_product_terms[1:3]
    for left in direct_aux_terms[:3]:
        if len(out) >= protected_aux_limit:
            break
        for right in combo_products:
            if len(out) >= protected_aux_limit:
                break
            if str(node_str(left.node)) == str(node_str(right.node)):
                continue
            _append_combo((left, right))
    for block in tail_products:
        if len(out) >= protected_aux_limit:
            break
        _append_combo((block,))
    for arity in range(1, max_arity + 1):
        if len(out) >= protected_aux_limit:
            break
        for combo in combinations(aux_term_rows, arity):
            if len(out) >= protected_aux_limit:
                break
            _append_combo(combo)

    affine_seed_rows = [
        block
        for block in list(seed_blocks or ())
        if isinstance(block, SeedBlock) and str(block.builder or "") == "affine"
    ]
    for block in affine_seed_rows:
        term_nodes = [node for node in list(dict(block.metadata or {}).get("term_nodes", []) or ()) if isinstance(node, tuple)]
        if not term_nodes:
            continue
        combo = tuple(
            SeedBlock(
                node=node,
                dim=block.dim,
                source="affine_term",
                builder="identity",
            )
            for node in term_nodes
        )
        _append_combo(combo)
        if len(out) >= int(limit):
            return out

    term_blocks = _carrier_candidates_for_spec(
        spec,
        target_dim=target_dim,
        anchor_dim=None,
        dim0_value=dim0_value,
        seed_blocks=seed_blocks,
        var_dims=var_dims,
        limit=max(2, int(limit)),
    )
    term_blocks = [block for block in term_blocks if block.node != ("const", 1.0)]
    if not term_blocks:
        return out
    combo_groups = [list(combinations(term_blocks, arity)) for arity in range(1, max_arity + 1)]
    while len(out) < int(limit):
        added = False
        for group in combo_groups:
            if not group:
                continue
            combo = tuple(group.pop(0))
            added = _append_combo(combo) or added
            if len(out) >= int(limit):
                break
        if not added:
            break
    return out


def _spec_scaffold_cap(
    spec: OperatorSpec,
    *,
    anchors_per_family: int,
    carrier_cap: int,
    affine_term_cap: int,
    remaining: int,
) -> int:
    operator_kind = str(getattr(spec, "operator_kind", "") or "")
    composition_mode = str(getattr(spec, "composition_mode", "") or "").strip().lower()
    if operator_kind == "affine_latent":
        base = max(2, min(12, int(affine_term_cap)))
    elif operator_kind in {"power_wrap", "quadratic_wrap"}:
        base = max(3, min(5, int(carrier_cap)))
    elif str(getattr(spec, "anchor_mode", "") or "") == "none":
        base = max(2, min(4, int(carrier_cap)))
    elif composition_mode == "prefactor":
        # Prefactor/envelope specs need a larger budget to reach monomial
        # seeds built from products (e.g., sqrt(x0*x1) as an envelope).
        base = max(4, min(12, int(anchors_per_family) * 2))
    else:
        base = max(2, min(6, int(anchors_per_family) + 2))
    return max(1, min(int(remaining), int(base)))


def _spec_carrier_cap(
    spec: OperatorSpec,
    *,
    anchors_per_family: int,
) -> int:
    operator_kind = str(getattr(spec, "operator_kind", "") or "")
    if operator_kind in {"power_wrap", "quadratic_wrap"}:
        return max(4, min(8, max(4, int(anchors_per_family) * 2)))
    if operator_kind == "fractional_head":
        return max(2, min(4, max(2, int(anchors_per_family) + 1)))
    if operator_kind == "affine_latent":
        return max(3, min(6, max(3, int(anchors_per_family) * 2)))
    return max(2, min(3, max(2, int(anchors_per_family))))


def _anchored_carrier_rows(
    spec: OperatorSpec,
    *,
    family_anchor_blocks_local: Sequence[SeedBlock],
    seed_blocks: Sequence[SeedBlock],
    target_dim: Any,
    dim0_value: Any,
    var_dims,
    limit: int,
) -> list[tuple[SeedBlock, list[SeedBlock | None]]]:
    anchor_candidates: list[SeedBlock] = []
    seen_anchor_nodes: set[str] = set()
    harmonic_prefactor = (
        str(getattr(spec, "operator_kind", "") or "").strip().lower() == "harmonic_wrap"
        and str(getattr(spec, "composition_mode", "") or "").strip().lower() == "prefactor"
    )

    def _add_anchor(block: SeedBlock | None) -> None:
        if not isinstance(block, SeedBlock):
            return
        anchor_node, anchor_dim = block.node, block.dim
        if not _anchor_allowed_for_spec(spec, anchor_dim=anchor_dim, dim0_value=dim0_value):
            return
        key = str(node_str(anchor_node))
        if key in seen_anchor_nodes:
            return
        seen_anchor_nodes.add(key)
        anchor_candidates.append(block)

    for block in family_anchor_blocks_local:
        _add_anchor(block)

    if str(getattr(spec, "operator_kind", "") or "").strip().lower() == "fractional_head":
        for block in list(seed_blocks or ()):
            if _is_wrapped_node(block.node):
                _add_anchor(block)

    recursive_anchor_limit = max(max(4, int(limit) * 3), len(tuple(family_anchor_blocks_local or ())), len(list(seed_blocks or ())))
    recursive_anchor_rows = sorted(list(seed_blocks or ()), key=_anchor_sort_key)
    for block in recursive_anchor_rows:
        if len(anchor_candidates) >= int(recursive_anchor_limit):
            break
        _add_anchor(block)
    if harmonic_prefactor:
        anchor_candidates.sort(key=_prefactor_harmonic_anchor_sort_key)

    rows: list[tuple[SeedBlock, list[SeedBlock | None]]] = []
    for anchor_block in anchor_candidates:
        anchor_node, anchor_dim = anchor_block.node, anchor_block.dim
        carrier_blocks_local = _carrier_candidates_for_spec(
            spec,
            target_dim=target_dim,
            anchor_dim=anchor_dim,
            dim0_value=dim0_value,
            seed_blocks=seed_blocks,
            var_dims=var_dims,
            limit=int(limit),
        )
        if callable(spec.carrier_dim_resolver) and not carrier_blocks_local:
            continue
        if not carrier_blocks_local:
            carrier_blocks_local = [None]
        distinct_carriers = [
            block
            for block in carrier_blocks_local
            if not (isinstance(block, SeedBlock) and block.node == anchor_node)
        ]
        same_carriers = [
            block
            for block in carrier_blocks_local
            if isinstance(block, SeedBlock) and block.node == anchor_node
        ]
        if distinct_carriers:
            carrier_blocks_local = [*distinct_carriers, *same_carriers]
        if harmonic_prefactor:
            distinct_sorted = sorted(
                distinct_carriers,
                key=lambda block: _prefactor_harmonic_carrier_sort_key(anchor_block, block),
            )
            same_sorted = sorted(
                same_carriers,
                key=lambda block: _prefactor_harmonic_carrier_sort_key(anchor_block, block),
            )
            carrier_blocks_local = [*distinct_sorted, *same_sorted]
        rows.append((anchor_block, list(carrier_blocks_local)))
    return rows


def _build_operator_bindings(
    spec: OperatorSpec,
    *,
    carrier_block: SeedBlock | None,
    anchor_block: SeedBlock | None,
) -> dict[str, Any]:
    bindings: dict[str, Any] = {}
    if spec.carrier_slot:
        bindings[str(spec.carrier_slot)] = (
            carrier_block if isinstance(carrier_block, SeedBlock) else ("const", 1.0)
        )
    if str(spec.operator_kind or "") == "quadratic_wrap" and isinstance(carrier_block, SeedBlock):
        base_blocks = _quadratic_base_bindings(carrier_block)
        if base_blocks:
            bindings["bases"] = base_blocks
    if spec.anchor_slot and isinstance(anchor_block, SeedBlock):
        bindings[str(spec.anchor_slot)] = anchor_block
    return bindings


def enumerate_operator_applications(
    *,
    families: Sequence[str] | None,
    nvars: int,
    y_dims,
    var_dims,
    pool_nodes: Sequence[tuple] | None,
    pool_dims: Sequence[Any] | None,
    anchors_per_family: int,
    max_scaffolds: int,
    basis_state=None,
    basis_state_beam: Sequence[Any] | None = None,
    aux_seed_blocks: Sequence[Any] | None = None,
    atom_library: Any = None,
    basis_seed_mode: str | None = None,
) -> list[OperatorApplication]:
    families_norm = normalize_families(families)
    if not families_norm or int(max_scaffolds) <= 0:
        return []

    dim0_value = dim0(var_dims)
    target_dim = tuple(y_dims) if isinstance(y_dims, (list, tuple)) else y_dims
    basis_seed_mode_token = _basis_seed_mode_token(basis_seed_mode)
    core_anchor_blocks = seed_anchor_blocks(
        nvars=nvars,
        pool_nodes=pool_nodes,
        pool_dims=pool_dims,
        var_dims=var_dims,
        max_count=max(1, int(anchors_per_family)),
    )
    use_augmented_seeds = basis_seed_mode_token in {"merged", "basis_augmented"}
    atom_lib = coerce_atom_library(atom_library)
    if use_augmented_seeds and atom_lib is not None:
        core_anchor_blocks = [
            enrich_seed_block_from_library(block, atom_lib)
            for block in list(core_anchor_blocks or ())
        ]
    anchor_blocks = list(core_anchor_blocks)
    aux_blocks: list[SeedBlock] = []
    if use_augmented_seeds:
        aux_cap = min(8, max(2, max(1, int(anchors_per_family)) * 2))
        aux_blocks = _sanitize_aux_seed_blocks(
            aux_seed_blocks,
            var_dims=var_dims,
            limit=int(aux_cap),
            atom_library=atom_library,
        )
        relation_blocks = list(
            seed_blocks_from_atom_relations(
                atom_lib,
                var_dims=var_dims,
                required_dim=target_dim,
                limit=max(2, min(8, max(1, int(anchors_per_family)) * 2)),
            )
        )
        aux_blocks = _merge_seed_blocks_unique(relation_blocks, aux_blocks, limit=int(aux_cap) + len(relation_blocks))
    if use_augmented_seeds:
        basis_cap = max(4, max(1, int(anchors_per_family)) * 3)
        anchor_limit = max(
            len(tuple(core_anchor_blocks or ())) + len(aux_blocks) + int(basis_cap),
            max(16, max(1, int(anchors_per_family)) * 8),
        )
        seed_blocks = _merge_seed_blocks_unique(core_anchor_blocks, aux_blocks)
        anchor_blocks = extend_seed_blocks_with_basis(
            seed_blocks,
            basis_state=basis_state,
            basis_state_beam=basis_state_beam,
            var_dims=var_dims,
            limit=anchor_limit,
            basis_limit=basis_cap,
            append_basis=True,
        )
        if atom_lib is not None or aux_blocks:
            anchor_blocks = sorted(anchor_blocks, key=_anchor_sort_key)
    recursive_seed_kwargs = _recursive_seed_pool_kwargs(
        families_norm,
        anchors_per_family=int(anchors_per_family),
    )
    quadratic_required_dims = _quadratic_required_expr_dims(
        families_norm,
        target_dim=target_dim,
        dim0_value=dim0_value,
        anchor_blocks=core_anchor_blocks,
    )
    core_carrier_blocks = build_recursive_seed_pool(
        core_anchor_blocks,
        **recursive_seed_kwargs,
        var_dims=var_dims,
        quadratic_required_dims=quadratic_required_dims,
    )
    carrier_blocks = list(core_carrier_blocks)
    if use_augmented_seeds:
        augmented_carrier_blocks = build_recursive_seed_pool(
            anchor_blocks,
            **recursive_seed_kwargs,
            var_dims=var_dims,
            quadratic_required_dims=quadratic_required_dims,
        )
        carrier_blocks = _merge_seed_blocks_unique(core_carrier_blocks, augmented_carrier_blocks)

    out: list[OperatorApplication] = []
    seen: set[str] = set()
    anchor_cap = max(0, int(anchors_per_family))
    affine_term_cap = max(3, min(8, max(2, int(anchors_per_family) * 3)))
    if use_augmented_seeds and aux_blocks:
        affine_term_cap = max(
            int(affine_term_cap),
            min(16, max(8, len(tuple(aux_blocks or ())))),
        )

    for family in families_norm:
        if len(out) >= int(max_scaffolds):
            break
        family_anchor_blocks_local = family_anchor_blocks(family, anchor_blocks, anchor_cap=anchor_cap)
        specs = family_operator_specs(family)
        for spec_idx, spec in enumerate(specs):
            if len(out) >= int(max_scaffolds):
                break
            remaining = max(1, int(max_scaffolds) - len(out))
            remaining_specs = max(1, int(len(specs) - int(spec_idx)))
            spec_carrier_cap = _spec_carrier_cap(spec, anchors_per_family=anchor_cap)
            spec_cap = _spec_scaffold_cap(
                spec,
                anchors_per_family=anchor_cap,
                carrier_cap=spec_carrier_cap,
                affine_term_cap=affine_term_cap,
                remaining=remaining,
            )
            fair_share = max(1, int((remaining + remaining_specs - 1) // remaining_specs))
            spec_cap = max(1, min(int(spec_cap), int(fair_share)))
            spec_start = len(out)
            if str(spec.operator_kind or "") == "affine_latent":
                term_combos = _affine_term_combos(
                    spec,
                    target_dim=target_dim,
                    dim0_value=dim0_value,
                    seed_blocks=carrier_blocks,
                    var_dims=var_dims,
                    limit=int(spec_cap),
                )
                for combo in term_combos:
                    if len(out) >= int(max_scaffolds) or (len(out) - spec_start) >= int(spec_cap):
                        break
                    bindings = {"terms": tuple(combo)}
                    term_nodes = tuple(block.node for block in combo)
                    parent_node = _render_affine_parent_node(term_nodes)
                    scaffold_id = f"affine:latent:{node_str(parent_node)}"
                    metadata = {
                        "form": spec.form,
                        "composition_mode": spec.composition_mode,
                        "operator": spec.operator_id,
                        "operator_kind": spec.operator_kind,
                        "composition_roles": list(spec.composition_roles or ()),
                        "carrier_role": str(spec.carrier_role or "carrier"),
                        "subset_role": str(spec.subset_role or ""),
                        "slot_bindings": binding_snapshot(bindings),
                    }
                    bound_closure = build_operator_bound_closure(
                        spec=spec,
                        family=family,
                        operator_id=spec.operator_id,
                        scaffold_id=scaffold_id,
                        parent_node=parent_node,
                        anchor_node=None,
                        bindings=bindings,
                        metadata=metadata,
                    )
                    if not validate_bound_closure_bindings(bound_closure, var_dims=var_dims, binding_values=bindings):
                        continue
                    append_operator_application(
                        out,
                        seen,
                        family=family,
                        operator_id=spec.operator_id,
                        scaffold_id=scaffold_id,
                        parent_node=parent_node,
                        hole_path=spec.hole_path,
                        target_mode=spec.target_mode,
                        bindings=bindings,
                        bound_closure=bound_closure,
                        metadata=metadata,
                        max_scaffolds=max_scaffolds,
                    )
                continue
            if spec.anchor_mode == "none":
                carrier_blocks_local = _carrier_candidates_for_spec(
                    spec,
                    target_dim=target_dim,
                    anchor_dim=None,
                    dim0_value=dim0_value,
                    seed_blocks=carrier_blocks,
                    var_dims=var_dims,
                    limit=int(spec_carrier_cap),
                )
                if callable(spec.carrier_dim_resolver) and not carrier_blocks_local:
                    continue
                if not carrier_blocks_local:
                    carrier_blocks_local = [None]
                for carrier_block in carrier_blocks_local:
                    if len(out) >= int(max_scaffolds) or (len(out) - spec_start) >= int(spec_cap):
                        break
                    carrier_node = carrier_block.node if isinstance(carrier_block, SeedBlock) else ("const", 1.0)
                    bindings = _build_operator_bindings(spec, carrier_block=carrier_block, anchor_block=None)
                    parent_node = _render_operator_template(
                        spec.parent_template,
                        carrier_node=carrier_node,
                        anchor_node=None,
                    )
                    metadata = {
                        "form": spec.form,
                        "composition_mode": spec.composition_mode,
                        "operator": spec.operator_id,
                        "operator_kind": spec.operator_kind,
                        "composition_roles": list(spec.composition_roles or ()),
                        "carrier_role": str(spec.carrier_role or "carrier"),
                    }
                    if spec.wrap_op is not None:
                        metadata["wrap_op"] = str(spec.wrap_op)
                    if spec.exponent is not None:
                        metadata["exponent"] = float(spec.exponent)
                    if bindings:
                        metadata["slot_bindings"] = binding_snapshot(bindings)
                    bound_closure = build_operator_bound_closure(
                        spec=spec,
                        family=family,
                        operator_id=spec.operator_id,
                        scaffold_id=spec.scaffold_id_builder(None, carrier_node),
                        parent_node=parent_node,
                        anchor_node=None,
                        bindings=bindings,
                        metadata=metadata,
                    )
                    if not validate_bound_closure_bindings(bound_closure, var_dims=var_dims, binding_values=bindings):
                        continue
                    append_operator_application(
                        out,
                        seen,
                        family=family,
                        operator_id=spec.operator_id,
                        scaffold_id=spec.scaffold_id_builder(None, carrier_node),
                        parent_node=parent_node,
                        hole_path=spec.hole_path,
                        target_mode=spec.target_mode,
                        bindings=bindings,
                        bound_closure=bound_closure,
                        metadata=metadata,
                        max_scaffolds=max_scaffolds,
                    )
                continue

            anchored_rows = _anchored_carrier_rows(
                spec,
                family_anchor_blocks_local=family_anchor_blocks_local,
                seed_blocks=carrier_blocks,
                target_dim=target_dim,
                dim0_value=dim0_value,
                var_dims=var_dims,
                limit=int(spec_carrier_cap),
            )
            max_carriers = max((len(carrier_rows) for _, carrier_rows in anchored_rows), default=0)
            for carrier_idx in range(max_carriers):
                if len(out) >= int(max_scaffolds) or (len(out) - spec_start) >= int(spec_cap):
                    break
                for anchor_block, carrier_rows in anchored_rows:
                    if len(out) >= int(max_scaffolds) or (len(out) - spec_start) >= int(spec_cap):
                        break
                    if carrier_idx >= len(carrier_rows):
                        continue
                    anchor_node = anchor_block.node
                    carrier_block = carrier_rows[carrier_idx]
                    carrier_node = carrier_block.node if isinstance(carrier_block, SeedBlock) else ("const", 1.0)
                    bindings = _build_operator_bindings(spec, carrier_block=carrier_block, anchor_block=anchor_block)
                    parent_node = _render_operator_template(
                        spec.parent_template,
                        carrier_node=carrier_node,
                        anchor_node=anchor_node,
                    )
                    metadata = {
                        "form": spec.form,
                        "composition_mode": spec.composition_mode,
                        "operator": spec.operator_id,
                        "operator_kind": spec.operator_kind,
                        "composition_roles": list(spec.composition_roles or ()),
                        "carrier_role": str(spec.carrier_role or "carrier"),
                        "anchor_role": str(spec.anchor_role or ""),
                    }
                    if spec.wrap_op is not None:
                        metadata["wrap_op"] = str(spec.wrap_op)
                    if spec.exponent is not None:
                        metadata["exponent"] = float(spec.exponent)
                    if bindings:
                        metadata["slot_bindings"] = binding_snapshot(bindings)
                    scaffold_id = spec.scaffold_id_builder(anchor_node, carrier_node)
                    bound_closure = build_operator_bound_closure(
                        spec=spec,
                        family=family,
                        operator_id=spec.operator_id,
                        scaffold_id=scaffold_id,
                        parent_node=parent_node,
                        anchor_node=anchor_node,
                        bindings=bindings,
                        metadata=metadata,
                    )
                    if not validate_bound_closure_bindings(bound_closure, var_dims=var_dims, binding_values=bindings):
                        continue
                    append_operator_application(
                        out,
                        seen,
                        family=family,
                        operator_id=spec.operator_id,
                        scaffold_id=scaffold_id,
                        parent_node=parent_node,
                        hole_path=spec.hole_path,
                        target_mode=spec.target_mode,
                        anchor_node=anchor_node,
                        bindings=bindings,
                        bound_closure=bound_closure,
                        metadata=metadata,
                        max_scaffolds=max_scaffolds,
                    )

    return out[: max(0, int(max_scaffolds))]

__all__ = [
    "enumerate_operator_applications",
    "family_anchor_rows",
    "normalize_families",
    "seed_anchor_nodes",
]
