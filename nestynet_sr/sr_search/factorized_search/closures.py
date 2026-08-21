# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Any, Mapping, Sequence

from .expr_ast import is_valid_node, node_str


def _valid_node(node: Any) -> tuple | None:
    if isinstance(node, tuple) and is_valid_node(node):
        return node
    return None


def _snapshot_value(value: Any) -> Any:
    node = _valid_node(value)
    if node is not None:
        return str(node_str(node))
    if isinstance(value, Mapping):
        return {str(k): _snapshot_value(v) for k, v in dict(value).items()}
    if isinstance(value, (list, tuple)):
        return [_snapshot_value(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _slot_policy(
    *,
    allowed_builders: Sequence[str] | None = None,
    disallow_builders: Sequence[str] | None = None,
    max_builder_depth: int | None = None,
    max_nonlinear_depth: int | None = None,
    max_product_arity: int | None = None,
    require_recursive: bool | None = None,
) -> dict[str, Any]:
    meta: dict[str, Any] = {}
    if allowed_builders:
        meta["allowed_builders"] = tuple(str(v) for v in tuple(allowed_builders) if str(v))
    if disallow_builders:
        meta["disallow_builders"] = tuple(str(v) for v in tuple(disallow_builders) if str(v))
    if max_builder_depth is not None:
        meta["max_builder_depth"] = int(max_builder_depth)
    if max_nonlinear_depth is not None:
        meta["max_nonlinear_depth"] = int(max_nonlinear_depth)
    if max_product_arity is not None:
        meta["max_product_arity"] = int(max_product_arity)
    if require_recursive is not None:
        meta["require_recursive"] = bool(require_recursive)
    return meta


def _simple_recursive_policy(
    *,
    allow_quadratic: bool = False,
    max_builder_depth: int = 3,
    max_nonlinear_depth: int = 2,
    max_product_arity: int = 3,
) -> dict[str, Any]:
    allowed = ["identity", "product", "monomial", "basis_head", "basis_atom", "basis_latent"]
    if allow_quadratic:
        allowed.append("quadratic")
    return _slot_policy(
        allowed_builders=tuple(allowed),
        max_builder_depth=max_builder_depth,
        max_nonlinear_depth=max_nonlinear_depth,
        max_product_arity=max_product_arity,
    )


@dataclass(frozen=True)
class SlotSpec:
    name: str
    role: str
    dim_rule: Any = None
    domain_rule: Any = None
    arity_cap: int | None = None
    reuse_policy: str = "allow"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": str(self.name),
            "role": str(self.role),
            "dim_rule": _snapshot_value(self.dim_rule),
            "domain_rule": _snapshot_value(self.domain_rule),
            "arity_cap": None if self.arity_cap is None else int(self.arity_cap),
            "reuse_policy": str(self.reuse_policy),
            "metadata": _snapshot_value(self.metadata),
        }


@dataclass(frozen=True)
class ClosureSpec:
    closure_id: str
    family: str
    head_solver: str
    slot_specs: tuple[SlotSpec, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "closure_id": str(self.closure_id),
            "family": str(self.family),
            "head_solver": str(self.head_solver),
            "slot_specs": [slot.to_dict() for slot in self.slot_specs],
            "metadata": _snapshot_value(self.metadata),
        }


@dataclass(frozen=True)
class ClosureDesign:
    fit_matrix: Any = None
    probe_matrix: Any = None
    materializer: str = "literal"
    materializer_payload: Mapping[str, Any] = field(default_factory=dict)
    payload: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        fit_shape = tuple(int(v) for v in getattr(self.fit_matrix, "shape", ()))
        probe_shape = tuple(int(v) for v in getattr(self.probe_matrix, "shape", ()))
        return {
            "fit_shape": list(fit_shape),
            "probe_shape": list(probe_shape),
            "materializer": str(self.materializer),
            "materializer_payload": _snapshot_value(self.materializer_payload),
            "payload": _snapshot_value(self.payload),
            "metadata": _snapshot_value(self.metadata),
        }


@dataclass(frozen=True)
class BoundClosure:
    spec: ClosureSpec
    bindings: Mapping[str, Any] = field(default_factory=dict)
    diagnostics: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "spec": self.spec.to_dict(),
            "bindings": _snapshot_value(self.bindings),
            "diagnostics": _snapshot_value(self.diagnostics),
            "metadata": _snapshot_value(self.metadata),
        }


def bound_closure_identity_payload(bound_closure: BoundClosure | None) -> dict[str, Any]:
    if not isinstance(bound_closure, BoundClosure):
        return {}
    bindings = dict(bound_closure.bindings or {})
    key_order: list[str] = [str(slot.name) for slot in tuple(bound_closure.spec.slot_specs or ())]
    for token in (
        "carrier",
        "anchor",
        "envelope",
        "companion",
        "companions",
        "numerator",
        "denominator",
        "numerator_terms",
        "denominator_terms",
        "bases",
        "terms",
    ):
        if token in bindings and token not in key_order:
            key_order.append(token)
    binding_payload: dict[str, Any] = {}
    for key in key_order:
        if key not in bindings:
            continue
        snap = _snapshot_value(bindings.get(key))
        if snap in (None, "", [], {}):
            continue
        binding_payload[str(key)] = snap
    meta = dict(bound_closure.metadata or {})
    meta_payload = {
        key: _snapshot_value(meta.get(key))
        for key in ("periodic_kind", "wrap_op", "power_kind", "quadratic_kind", "exponent", "form")
        if key in meta
    }
    return {
        "closure_id": str(bound_closure.spec.closure_id),
        "family": str(bound_closure.spec.family),
        "head_solver": str(bound_closure.spec.head_solver),
        "bindings": binding_payload,
        "metadata": meta_payload,
    }


def bound_closure_identity_key(bound_closure: BoundClosure | None) -> str:
    payload = bound_closure_identity_payload(bound_closure)
    if not payload:
        return ""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def make_bound_closure(
    *,
    closure_id: str,
    family: str,
    head_solver: str,
    slot_specs: Sequence[SlotSpec] | None = None,
    bindings: Mapping[str, Any] | None = None,
    diagnostics: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> BoundClosure:
    return BoundClosure(
        spec=ClosureSpec(
            closure_id=str(closure_id),
            family=str(family),
            head_solver=str(head_solver),
            slot_specs=tuple(slot_specs or ()),
            metadata=dict(metadata or {}),
        ),
        bindings=dict(bindings or {}),
        diagnostics=dict(diagnostics or {}),
        metadata=dict(metadata or {}),
    )


def make_structured_head_closure(
    *,
    closure_id: str,
    family: str,
    head_solver: str,
    scaffold_id: str,
    slot_specs: Sequence[SlotSpec] | None = None,
    bindings: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
    diagnostics: Mapping[str, Any] | None = None,
) -> BoundClosure:
    meta = {"scaffold_id": str(scaffold_id), **dict(metadata or {})}
    diag = {"scaffold_id": str(scaffold_id), **dict(diagnostics or {})}
    return make_bound_closure(
        closure_id=str(closure_id),
        family=str(family),
        head_solver=str(head_solver),
        slot_specs=slot_specs,
        bindings=bindings,
        diagnostics=diag,
        metadata=meta,
    )


def make_wrapped_linear_head_closure(
    *,
    scaffold_id: str,
    family: str,
    wrap_kind: str,
    wrap_op: str,
    hole_node: tuple,
    feature_node: tuple | None = None,
    anchor_node: tuple | None = None,
    carrier_dim_rule: Any = "dimless",
    carrier_domain_rule: Any = None,
    anchor_role: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> BoundClosure:
    family_token = str(family or "").strip().lower() or "generic"
    kind_token = str(wrap_kind or "").strip().lower() or "base"
    wrap_token = str(wrap_op or "").strip().lower() or family_token
    anchor_role_token = str(anchor_role or ("envelope" if kind_token == "mul" else "companion"))
    slots = [
        SlotSpec(
            name="carrier",
            role="carrier",
            dim_rule=carrier_dim_rule,
            domain_rule=carrier_domain_rule,
            metadata=_simple_recursive_policy(
                allow_quadratic=False,
                max_builder_depth=3,
                max_nonlinear_depth=1,
                max_product_arity=3,
            ),
        )
    ]
    if isinstance(anchor_node, tuple):
        slots.append(
            SlotSpec(
                name="anchor",
                role=anchor_role_token,
                reuse_policy="distinct_from:carrier",
                metadata=_simple_recursive_policy(
                    allow_quadratic=False,
                    max_builder_depth=3,
                    max_nonlinear_depth=2,
                    max_product_arity=3,
                ),
            )
        )
    feature = _valid_node(feature_node) or (wrap_token, hole_node)
    return make_structured_head_closure(
        closure_id=f"{family_token}:{kind_token}:linear_head",
        family=family_token,
        head_solver="linear",
        scaffold_id=str(scaffold_id),
        slot_specs=tuple(slots),
        bindings={
            "carrier": hole_node,
            "feature": feature,
            "anchor": anchor_node,
        },
        metadata={
            "wrap_op": wrap_token,
            f"{family_token}_kind": kind_token,
            **dict(metadata or {}),
        },
    )


def make_direct_periodic_closure(
    *,
    scaffold_id: str,
    periodic_kind: str,
    hole_node: tuple,
    feature_node: tuple,
    anchor_node: tuple | None,
    envelope_node: tuple | None = None,
    companion_nodes: Sequence[tuple] | None = None,
    harmonic_feature_nodes: Sequence[tuple] | None = None,
    expr: tuple | None = None,
) -> BoundClosure:
    companion_list = tuple(
        node for node in list(companion_nodes or ()) if _valid_node(node) is not None
    )
    harmonic_feature_list = tuple(
        node for node in list(harmonic_feature_nodes or ()) if _valid_node(node) is not None
    )
    slot_specs = [
        SlotSpec(
            name="carrier",
            role="carrier",
            dim_rule="dimless",
            metadata=_simple_recursive_policy(
                allow_quadratic=False,
                max_builder_depth=2,
                max_nonlinear_depth=1,
                max_product_arity=3,
            ),
        )
    ]
    if _valid_node(envelope_node) is not None:
        slot_specs.append(
            SlotSpec(
                name="envelope",
                role="envelope",
                reuse_policy="distinct_from:carrier",
                metadata=_simple_recursive_policy(
                    allow_quadratic=False,
                    max_builder_depth=3,
                    max_nonlinear_depth=2,
                    max_product_arity=3,
                ),
            )
        )
    if companion_list:
        slot_specs.append(
            SlotSpec(
                name="companions",
                role="companion",
                arity_cap=int(len(companion_list)),
                reuse_policy="distinct_from:carrier,envelope",
                metadata=_simple_recursive_policy(
                    allow_quadratic=False,
                    max_builder_depth=3,
                    max_nonlinear_depth=2,
                    max_product_arity=3,
                ),
            )
        )
    return make_structured_head_closure(
        closure_id=f"periodic:{str(periodic_kind)}:harmonic_linear",
        family="periodic",
        head_solver="harmonic_linear",
        scaffold_id=str(scaffold_id),
        slot_specs=tuple(slot_specs),
        bindings={
            "carrier": hole_node,
            "feature": feature_node,
            "envelope": envelope_node,
            "companion": anchor_node,
            "companions": companion_list,
            "harmonic_features": harmonic_feature_list,
            "expr": expr,
        },
        metadata={"periodic_kind": str(periodic_kind)},
    )


def make_direct_linear_wrap_closure(
    *,
    scaffold_id: str,
    family: str,
    wrap_kind: str,
    wrap_op: str,
    hole_node: tuple,
    feature_node: tuple | None = None,
    anchor_node: tuple | None = None,
    carrier_dim_rule: Any = "dimless",
    carrier_domain_rule: Any = None,
    anchor_role: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> BoundClosure:
    return make_wrapped_linear_head_closure(
        scaffold_id=str(scaffold_id),
        family=str(family),
        wrap_kind=str(wrap_kind),
        wrap_op=str(wrap_op),
        hole_node=hole_node,
        feature_node=feature_node,
        anchor_node=anchor_node,
        carrier_dim_rule=carrier_dim_rule,
        carrier_domain_rule=carrier_domain_rule,
        anchor_role=anchor_role,
        metadata=metadata,
    )


def make_direct_exp_closure(
    *,
    scaffold_id: str,
    exp_kind: str,
    hole_node: tuple,
    feature_node: tuple,
    anchor_node: tuple | None = None,
) -> BoundClosure:
    return make_direct_linear_wrap_closure(
        scaffold_id=scaffold_id,
        family="exp",
        wrap_kind=str(exp_kind),
        wrap_op="exp",
        hole_node=hole_node,
        feature_node=feature_node,
        anchor_node=anchor_node,
    )


def make_direct_affine_closure(
    *,
    scaffold_id: str,
    term_nodes: Sequence[tuple],
) -> BoundClosure:
    valid_terms = tuple(node for node in list(term_nodes or ()) if _valid_node(node) is not None)
    slots = ()
    if valid_terms:
        slots = (
            SlotSpec(
                name="terms",
                role="term",
                arity_cap=max(1, int(len(valid_terms))),
                reuse_policy="pairwise_distinct",
                metadata=_simple_recursive_policy(
                    allow_quadratic=True,
                    max_builder_depth=3,
                    max_nonlinear_depth=2,
                    max_product_arity=3,
                ),
            ),
        )
    return make_structured_head_closure(
        closure_id="affine:latent_linear_head",
        family="affine",
        head_solver="linear",
        scaffold_id=str(scaffold_id),
        slot_specs=slots,
        bindings={
            "terms": valid_terms,
        },
        metadata={"form": "affine_latent"},
    )


def make_direct_log_closure(
    *,
    scaffold_id: str,
    log_kind: str,
    hole_node: tuple,
    feature_node: tuple,
    anchor_node: tuple | None = None,
) -> BoundClosure:
    return make_direct_linear_wrap_closure(
        scaffold_id=scaffold_id,
        family="log",
        wrap_kind=str(log_kind),
        wrap_op="log",
        hole_node=hole_node,
        feature_node=feature_node,
        anchor_node=anchor_node,
        carrier_domain_rule="positive_output",
        anchor_role="companion",
    )


def make_direct_rational_closure(
    *,
    scaffold_id: str,
    u_node: tuple,
    v_node: tuple,
) -> BoundClosure:
    return make_structured_head_closure(
        closure_id="rational:affine_fractional_head",
        family="rational",
        head_solver="fractional_linear",
        scaffold_id=str(scaffold_id),
        slot_specs=(
            SlotSpec(
                name="numerator",
                role="carrier",
                metadata=_simple_recursive_policy(
                    allow_quadratic=True,
                    max_builder_depth=3,
                    max_nonlinear_depth=2,
                    max_product_arity=3,
                ),
            ),
            SlotSpec(
                name="denominator",
                role="denominator",
                dim_rule="dimless",
                reuse_policy="distinct_from:numerator",
                metadata=_simple_recursive_policy(
                    allow_quadratic=True,
                    max_builder_depth=3,
                    max_nonlinear_depth=2,
                    max_product_arity=3,
                ),
            ),
        ),
        bindings={
            "numerator": u_node,
            "denominator": v_node,
        },
        metadata={"form": "rational_affine"},
    )


def make_multi_term_rational_closure(
    *,
    scaffold_id: str,
    u_nodes: Sequence[tuple],
    v_nodes: Sequence[tuple],
) -> BoundClosure:
    """Create a BoundClosure for (a0 + sum a_i u_i) / (1 + sum b_j v_j)."""
    valid_u = tuple(node for node in list(u_nodes or ()) if _valid_node(node) is not None)
    valid_v = tuple(node for node in list(v_nodes or ()) if _valid_node(node) is not None)
    slots = []
    if valid_u:
        slots.append(
            SlotSpec(
                name="numerator_terms",
                role="carrier",
                arity_cap=max(1, int(len(valid_u))),
                reuse_policy="pairwise_distinct",
                metadata=_simple_recursive_policy(
                    allow_quadratic=True,
                    max_builder_depth=3,
                    max_nonlinear_depth=2,
                    max_product_arity=3,
                ),
            )
        )
    if valid_v:
        slots.append(
            SlotSpec(
                name="denominator_terms",
                role="denominator",
                dim_rule="dimless",
                arity_cap=max(1, int(len(valid_v))),
                reuse_policy="pairwise_distinct",
                metadata=_simple_recursive_policy(
                    allow_quadratic=True,
                    max_builder_depth=3,
                    max_nonlinear_depth=2,
                    max_product_arity=3,
                ),
            )
        )
    return make_structured_head_closure(
        closure_id="rational:multi_term_fractional_head",
        family="rational",
        head_solver="multi_term_fractional",
        scaffold_id=str(scaffold_id),
        slot_specs=tuple(slots),
        bindings={
            "numerator_terms": valid_u,
            "denominator_terms": valid_v,
        },
        metadata={"form": "multi_term_rational"},
    )


def make_direct_quadratic_closure(
    *,
    scaffold_id: str,
    quadratic_kind: str,
    base_nodes: Sequence[tuple],
    anchor_node: tuple | None = None,
) -> BoundClosure:
    valid_bases = tuple(node for node in list(base_nodes or ()) if _valid_node(node) is not None)
    slots = [
        SlotSpec(
            name="bases",
            role="carrier",
            arity_cap=max(1, int(len(valid_bases) or 1)),
            reuse_policy="pairwise_distinct",
            metadata=_slot_policy(
                allowed_builders=("identity", "product", "basis_head", "basis_atom", "basis_latent"),
                max_builder_depth=2,
                max_nonlinear_depth=1,
                max_product_arity=2,
            ),
        )
    ]
    if _valid_node(anchor_node) is not None:
        slots.append(
            SlotSpec(
                name="anchor",
                role="envelope",
                reuse_policy="distinct_from:bases",
                metadata=_simple_recursive_policy(
                    allow_quadratic=False,
                    max_builder_depth=3,
                    max_nonlinear_depth=2,
                    max_product_arity=3,
                ),
            )
        )
    return make_structured_head_closure(
        closure_id=f"quadratic:{str(quadratic_kind)}:sqrt_head",
        family="quadratic",
        head_solver="quadratic_sqrt",
        scaffold_id=str(scaffold_id),
        slot_specs=tuple(slots),
        bindings={
            "bases": valid_bases,
            "anchor": anchor_node,
        },
        metadata={"quadratic_kind": str(quadratic_kind)},
    )


def make_direct_power_closure(
    *,
    scaffold_id: str,
    power_kind: str,
    exponent: float,
    hole_node: tuple,
    anchor_node: tuple | None = None,
) -> BoundClosure:
    domain_rule = None
    try:
        exponent_f = float(exponent)
    except Exception:
        exponent_f = 0.0
    if exponent_f in {0.5, 2.0}:
        domain_rule = "nonnegative_output" if exponent_f == 0.5 else None
    elif exponent_f in {-0.5, -1.0, -2.0}:
        # Accept nonnegative carriers (not just strictly positive) — the
        # power head fits (a0 + a1*h)^p and checks positivity on the actual
        # fitted inner values, so the raw carrier can be nonnegative.
        # This unlocks ratio-square carriers like sqr(x0/x1) for Lorentz.
        domain_rule = "nonnegative_output"
    slots = [
        SlotSpec(
            name="carrier",
            role="carrier",
            domain_rule=domain_rule,
            metadata=_simple_recursive_policy(
                allow_quadratic=True,
                max_builder_depth=3,
                max_nonlinear_depth=2,
                max_product_arity=3,
            ),
        )
    ]
    if _valid_node(anchor_node) is not None:
        slots.append(
            SlotSpec(
                name="anchor",
                role="envelope",
                reuse_policy="distinct_from:carrier",
                metadata=_simple_recursive_policy(
                    allow_quadratic=False,
                    max_builder_depth=3,
                    max_nonlinear_depth=2,
                    max_product_arity=3,
                ),
            )
        )
    return make_structured_head_closure(
        closure_id=f"power:{str(power_kind)}:discrete_power",
        family="power",
        head_solver="discrete_power",
        scaffold_id=str(scaffold_id),
        slot_specs=tuple(slots),
        bindings={
            "carrier": hole_node,
            "anchor": anchor_node,
        },
        metadata={
            "power_kind": str(power_kind),
            "exponent": float(exponent),
        },
    )


def bound_closure_from_closure_candidate(
    *,
    family: str,
    scaffold_id: str,
    expr: tuple | None,
    anchor_node: tuple | None,
    scaffold_metadata: Mapping[str, Any] | None,
    direct_metadata: Mapping[str, Any] | None = None,
) -> BoundClosure:
    direct_meta = dict(direct_metadata or {})
    scaffold_meta = dict(scaffold_metadata or {})
    fam = str(family or "").strip().lower()
    form = str(scaffold_meta.get("form", "") or "").strip().lower()
    if fam == "periodic":
        periodic_kind = str(direct_meta.get("feature_kind", "cos") or "cos")
        hole_node = _valid_node(direct_meta.get("hole_node", None)) or _valid_node(expr) or ("const", 1.0)
        feature_node = _valid_node(direct_meta.get("feature_node", None)) or (
            (periodic_kind, hole_node) if _valid_node(hole_node) is not None else ("const", 1.0)
        )
        companion_nodes = []
        for raw_node in list(direct_meta.get("companion_nodes", []) or ()):
            node = _valid_node(raw_node)
            if node is not None:
                companion_nodes.append(node)
        harmonic_feature_nodes = []
        for raw_node in list(direct_meta.get("harmonic_feature_nodes", []) or ()):
            node = _valid_node(raw_node)
            if node is not None:
                harmonic_feature_nodes.append(node)
        return make_direct_periodic_closure(
            scaffold_id=scaffold_id,
            periodic_kind=periodic_kind,
            hole_node=hole_node,
            feature_node=feature_node,
            anchor_node=_valid_node(anchor_node),
            envelope_node=_valid_node(direct_meta.get("envelope_node", None)),
            companion_nodes=tuple(companion_nodes),
            harmonic_feature_nodes=tuple(harmonic_feature_nodes),
            expr=_valid_node(expr),
        )
    if fam == "exp":
        exp_kind = "base"
        if form == "exp_add":
            exp_kind = "add"
        elif form == "exp_mul":
            exp_kind = "mul"
        hole_node = _valid_node(direct_meta.get("hole_node", None)) or ("const", 1.0)
        feature_node = _valid_node(direct_meta.get("feature_node", None)) or ("exp", hole_node)
        return make_direct_linear_wrap_closure(
            scaffold_id=scaffold_id,
            family="exp",
            wrap_kind=exp_kind,
            wrap_op="exp",
            hole_node=hole_node,
            feature_node=feature_node,
            anchor_node=_valid_node(anchor_node),
        )
    if fam == "affine":
        term_nodes = []
        for raw_node in list(direct_meta.get("term_nodes", []) or ()):
            node = _valid_node(raw_node)
            if node is not None:
                term_nodes.append(node)
        if not term_nodes:
            expr_node = _valid_node(expr)
            if expr_node is not None:
                term_nodes.append(expr_node)
        return make_direct_affine_closure(
            scaffold_id=scaffold_id,
            term_nodes=tuple(term_nodes),
        )
    if fam == "log":
        log_kind = "add" if form == "log_add" else "base"
        hole_node = _valid_node(direct_meta.get("hole_node", None)) or ("const", 1.0)
        feature_node = _valid_node(direct_meta.get("feature_node", None)) or ("log", hole_node)
        return make_direct_linear_wrap_closure(
            scaffold_id=scaffold_id,
            family="log",
            wrap_kind=log_kind,
            wrap_op="log",
            hole_node=hole_node,
            feature_node=feature_node,
            anchor_node=_valid_node(anchor_node),
            carrier_domain_rule="positive_output",
            anchor_role="companion",
        )
    if fam == "rational":
        if form == "multi_term_rational":
            u_nodes = []
            for raw_node in list(direct_meta.get("u_nodes", []) or ()):
                node = _valid_node(raw_node)
                if node is not None:
                    u_nodes.append(node)
            v_nodes = []
            for raw_node in list(direct_meta.get("v_nodes", []) or ()):
                node = _valid_node(raw_node)
                if node is not None:
                    v_nodes.append(node)
            return make_multi_term_rational_closure(
                scaffold_id=scaffold_id,
                u_nodes=tuple(u_nodes),
                v_nodes=tuple(v_nodes),
            )
        u_node = _valid_node(direct_meta.get("u_node", None)) or ("const", 0.0)
        v_node = _valid_node(direct_meta.get("v_node", None)) or ("const", 0.0)
        return make_direct_rational_closure(
            scaffold_id=scaffold_id,
            u_node=u_node,
            v_node=v_node,
        )
    if fam == "quadratic":
        base_nodes = []
        for raw_node in list(direct_meta.get("quadratic_base_nodes", []) or ()):
            node = _valid_node(raw_node)
            if node is not None:
                base_nodes.append(node)
        return make_direct_quadratic_closure(
            scaffold_id=scaffold_id,
            quadratic_kind=str(direct_meta.get("quadratic_kind", "sqrt") or "sqrt"),
            base_nodes=tuple(base_nodes),
            anchor_node=_valid_node(direct_meta.get("anchor_node", None)) or _valid_node(anchor_node),
        )
    if fam == "power":
        hole_node = _valid_node(direct_meta.get("hole_node", None)) or ("const", 1.0)
        exponent = float(direct_meta.get("power_exponent", 0.5) or 0.5)
        return make_direct_power_closure(
            scaffold_id=scaffold_id,
            power_kind=str(direct_meta.get("power_kind", "sqrt") or "sqrt"),
            exponent=exponent,
            hole_node=hole_node,
            anchor_node=_valid_node(direct_meta.get("anchor_node", None)) or _valid_node(anchor_node),
        )
    return make_bound_closure(
        closure_id=f"{fam or 'unknown'}:literal",
        family=str(family),
        head_solver="unknown",
        slot_specs=(),
        bindings={"expr": _valid_node(expr)},
        diagnostics={"scaffold_id": str(scaffold_id)},
        metadata={"scaffold_id": str(scaffold_id), "form": form},
    )


__all__ = [
    "BoundClosure",
    "ClosureDesign",
    "ClosureSpec",
    "SlotSpec",
    "bound_closure_identity_key",
    "bound_closure_identity_payload",
    "bound_closure_from_closure_candidate",
    "make_bound_closure",
    "make_structured_head_closure",
    "make_wrapped_linear_head_closure",
    "make_direct_affine_closure",
    "make_direct_linear_wrap_closure",
    "make_direct_exp_closure",
    "make_direct_log_closure",
    "make_direct_power_closure",
    "make_direct_periodic_closure",
    "make_direct_quadratic_closure",
    "make_direct_rational_closure",
    "make_multi_term_rational_closure",
]
