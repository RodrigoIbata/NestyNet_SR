# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from .common import dim_add, dim_scale, dim_sub


CarrierDimResolver = Callable[[Any, Any, Any], Any]
ScaffoldIdBuilder = Callable[[tuple | None, tuple | None], str]


@dataclass(frozen=True)
class OperatorSpec:
    family: str
    operator_kind: str
    operator_id: str
    scaffold_id_builder: ScaffoldIdBuilder
    parent_template: tuple
    hole_path: tuple[int, ...]
    target_mode: str
    form: str
    composition_mode: str = "base"
    wrap_op: str | None = None
    exponent: float | None = None
    carrier_domain_rule: Any = None
    anchor_mode: str = "none"
    anchor_dim_rule: str | None = None
    carrier_slot: str | None = "carrier"
    anchor_slot: str | None = None
    carrier_dim_resolver: CarrierDimResolver | None = None
    subset_slot: str | None = None
    subset_max_arity: int = 1
    carrier_role: str = "carrier"
    anchor_role: str | None = None
    subset_role: str | None = None
    composition_roles: tuple[str, ...] = ()


def _static_scaffold_id(value: str) -> ScaffoldIdBuilder:
    def _builder(anchor_node: tuple | None, carrier_node: tuple | None = None) -> str:
        from ..expr_ast import node_str

        if isinstance(carrier_node, tuple):
            return f"{value}:{node_str(carrier_node)}"
        return str(value)

    return _builder


def _anchored_scaffold_id(prefix: str) -> ScaffoldIdBuilder:
    def _builder(anchor_node: tuple | None, carrier_node: tuple | None = None) -> str:
        from ..expr_ast import node_str

        parts = [prefix]
        if isinstance(anchor_node, tuple):
            parts.append(node_str(anchor_node))
        if isinstance(carrier_node, tuple):
            parts.append(node_str(carrier_node))
        return ":".join(parts) if len(parts) > 1 else str(prefix)

    return _builder


def _numerator_dim(target_dim: Any, anchor_dim: Any, _dim0_value: Any) -> Any:
    return dim_add(target_dim, anchor_dim)


def _denominator_dim(target_dim: Any, anchor_dim: Any, _dim0_value: Any) -> Any:
    return dim_sub(anchor_dim, target_dim)


def _dim0_resolver(_target_dim: Any, _anchor_dim: Any, dim0_value: Any) -> Any:
    return dim0_value


def _target_dim_resolver(target_dim: Any, _anchor_dim: Any, _dim0_value: Any) -> Any:
    return target_dim


def _power_base_dim(exponent: float) -> CarrierDimResolver:
    def _resolve(target_dim: Any, _anchor_dim: Any, _dim0_value: Any) -> Any:
        if target_dim is None:
            return None
        return dim_scale(target_dim, 1.0 / float(exponent))

    return _resolve


def _quadratic_latent_dim(target_dim: Any, _anchor_dim: Any, _dim0_value: Any) -> Any:
    if target_dim is None:
        return None
    return dim_scale(target_dim, 2.0)


def _quadratic_latent_mul_dim(target_dim: Any, anchor_dim: Any, _dim0_value: Any) -> Any:
    core_dim = dim_sub(target_dim, anchor_dim)
    if core_dim is None:
        return None
    return dim_scale(core_dim, 2.0)


def _composition_roles_for_mode(mode: str) -> tuple[str, ...]:
    token = str(mode or "").strip().lower()
    return {
        "base": ("wrapper",),
        "companion": ("wrapper", "companion"),
        "prefactor": ("wrapper", "prefactor"),
        "latent": ("affine_latent",),
        "denominator_companion": ("fractional_head", "denominator_companion"),
        "numerator_companion": ("fractional_head", "numerator_companion"),
        "fractional": ("fractional_head",),
    }.get(token, ())


def _make_composed_wrap_spec(
    *,
    family: str,
    operator_kind: str,
    operator_id: str,
    scaffold_id_builder: ScaffoldIdBuilder,
    parent_template: tuple,
    hole_path: tuple[int, ...],
    target_mode: str,
    form: str,
    composition_mode: str,
    wrap_op: str | None = None,
    exponent: float | None = None,
    carrier_dim_resolver: CarrierDimResolver | None = None,
    carrier_domain_rule: Any = None,
    anchor_mode: str = "none",
    anchor_dim_rule: str | None = None,
    anchor_slot: str | None = None,
    carrier_slot: str | None = "carrier",
    carrier_role: str = "carrier",
    anchor_role: str | None = None,
) -> OperatorSpec:
    return OperatorSpec(
        family=str(family),
        operator_kind=str(operator_kind),
        operator_id=str(operator_id),
        scaffold_id_builder=scaffold_id_builder,
        parent_template=parent_template,
        hole_path=hole_path,
        target_mode=str(target_mode),
        form=str(form),
        composition_mode=str(composition_mode),
        wrap_op=None if wrap_op is None else str(wrap_op),
        exponent=None if exponent is None else float(exponent),
        carrier_domain_rule=carrier_domain_rule,
        anchor_mode=str(anchor_mode),
        anchor_dim_rule=anchor_dim_rule,
        carrier_slot=carrier_slot,
        anchor_slot=anchor_slot,
        carrier_dim_resolver=carrier_dim_resolver,
        carrier_role=str(carrier_role),
        anchor_role=None if anchor_role is None else str(anchor_role),
        composition_roles=_composition_roles_for_mode(str(composition_mode)),
    )


def _power_mul_dim(exponent: float) -> CarrierDimResolver:
    def _resolve(target_dim: Any, anchor_dim: Any, _dim0_value: Any) -> Any:
        core_dim = dim_sub(target_dim, anchor_dim)
        if core_dim is None:
            return None
        return dim_scale(core_dim, 1.0 / float(exponent))

    return _resolve


def _make_harmonic_spec(
    *,
    periodic_kind: str,
    variant: str,
) -> OperatorSpec:
    token = str(periodic_kind).strip().lower()
    variant_token = str(variant).strip().lower()
    if variant_token == "base":
        return _make_composed_wrap_spec(
            family="periodic",
            operator_kind="harmonic_wrap",
            operator_id=f"periodic:{token}_base",
            scaffold_id_builder=_static_scaffold_id(f"periodic:{token}"),
            parent_template=(token, "__CARRIER__"),
            hole_path=(1,),
            target_mode="robust",
            form=f"{token}_base",
            composition_mode="base",
            wrap_op=token,
            carrier_dim_resolver=_dim0_resolver,
        )
    if variant_token == "add":
        return _make_composed_wrap_spec(
            family="periodic",
            operator_kind="harmonic_wrap",
            operator_id=f"periodic:{token}_add",
            scaffold_id_builder=_anchored_scaffold_id(f"periodic:{token}_add"),
            parent_template=("add", (token, "__CARRIER__"), "__ANCHOR__"),
            hole_path=(1, 1),
            target_mode="robust",
            form=f"{token}_add",
            composition_mode="companion",
            wrap_op=token,
            anchor_mode="per_anchor",
            anchor_dim_rule="dim0",
            anchor_slot="companion",
            carrier_dim_resolver=_dim0_resolver,
            anchor_role="companion",
        )
    if variant_token == "mul":
        return _make_composed_wrap_spec(
            family="periodic",
            operator_kind="harmonic_wrap",
            operator_id=f"periodic:{token}_mul",
            scaffold_id_builder=_anchored_scaffold_id(f"periodic:{token}_mul"),
            parent_template=("mul", (token, "__CARRIER__"), "__ANCHOR__"),
            hole_path=(1, 1),
            target_mode="robust",
            form=f"{token}_mul",
            composition_mode="prefactor",
            wrap_op=token,
            anchor_mode="per_anchor",
            anchor_slot="envelope",
            carrier_dim_resolver=_dim0_resolver,
            anchor_role="prefactor",
        )
    raise ValueError(f"unsupported harmonic variant: {variant}")


def _make_unary_wrap_spec(
    *,
    family: str,
    wrap_op: str,
    variant: str,
) -> OperatorSpec:
    family_token = str(family).strip().lower()
    wrap_token = str(wrap_op).strip().lower()
    variant_token = str(variant).strip().lower()
    if variant_token == "base":
        return _make_composed_wrap_spec(
            family=family_token,
            operator_kind="unary_wrap",
            operator_id=f"{family_token}:base",
            scaffold_id_builder=_static_scaffold_id(f"{family_token}:base"),
            parent_template=(wrap_token, "__CARRIER__"),
            hole_path=(1,),
            target_mode="robust",
            form=f"{family_token}_base",
            composition_mode="base",
            wrap_op=wrap_token,
            carrier_dim_resolver=_dim0_resolver,
            carrier_domain_rule="positive_output" if family_token == "log" else None,
        )
    if variant_token == "add":
        return _make_composed_wrap_spec(
            family=family_token,
            operator_kind="anchored_unary_wrap",
            operator_id=f"{family_token}:add",
            scaffold_id_builder=_anchored_scaffold_id(f"{family_token}:add"),
            parent_template=("add", (wrap_token, "__CARRIER__"), "__ANCHOR__"),
            hole_path=(1, 1),
            target_mode="robust",
            form=f"{family_token}_add",
            composition_mode="companion",
            wrap_op=wrap_token,
            anchor_mode="per_anchor",
            anchor_dim_rule="dim0",
            anchor_slot="anchor",
            carrier_dim_resolver=_dim0_resolver,
            carrier_domain_rule="positive_output" if family_token == "log" else None,
            anchor_role="companion",
        )
    if variant_token == "mul":
        return _make_composed_wrap_spec(
            family=family_token,
            operator_kind="anchored_unary_wrap",
            operator_id=f"{family_token}:mul",
            scaffold_id_builder=_anchored_scaffold_id(f"{family_token}:mul"),
            parent_template=("mul", (wrap_token, "__CARRIER__"), "__ANCHOR__"),
            hole_path=(1, 1),
            target_mode="robust",
            form=f"{family_token}_mul",
            composition_mode="prefactor",
            wrap_op=wrap_token,
            anchor_mode="per_anchor",
            anchor_slot="anchor",
            carrier_dim_resolver=_dim0_resolver,
            carrier_domain_rule="positive_output" if family_token == "log" else None,
            anchor_role="prefactor",
        )
    raise ValueError(f"unsupported unary-wrap variant: {variant}")


def _make_fractional_spec(*, variant: str) -> OperatorSpec:
    variant_token = str(variant).strip().lower()
    if variant_token == "affine":
        return OperatorSpec(
            family="rational",
            operator_kind="fractional_head",
            operator_id="rational:affine",
            scaffold_id_builder=_static_scaffold_id("rational:affine"),
            parent_template=("div", ("const", 1.0), ("const", 1.0)),
            hole_path=(),
            target_mode="full",
            form="rational_affine",
            composition_mode="fractional",
            carrier_slot=None,
            composition_roles=_composition_roles_for_mode("fractional"),
        )
    if variant_token == "num_over_anchor":
        return OperatorSpec(
            family="rational",
            operator_kind="fractional_head",
            operator_id="rational:num_over_anchor",
            scaffold_id_builder=_anchored_scaffold_id("rational:num"),
            parent_template=("div", "__CARRIER__", "__ANCHOR__"),
            hole_path=(1,),
            target_mode="full",
            form="num_over_anchor",
            composition_mode="denominator_companion",
            anchor_mode="per_anchor",
            carrier_slot="numerator",
            anchor_slot="anchor",
            carrier_dim_resolver=_numerator_dim,
            carrier_role="numerator",
            anchor_role="denominator_companion",
            composition_roles=_composition_roles_for_mode("denominator_companion"),
        )
    if variant_token == "anchor_over_den":
        return OperatorSpec(
            family="rational",
            operator_kind="fractional_head",
            operator_id="rational:anchor_over_den",
            scaffold_id_builder=_anchored_scaffold_id("rational:den"),
            parent_template=("div", "__ANCHOR__", "__CARRIER__"),
            hole_path=(2,),
            target_mode="full",
            form="anchor_over_den",
            composition_mode="numerator_companion",
            anchor_mode="per_anchor",
            carrier_slot="denominator",
            anchor_slot="anchor",
            carrier_dim_resolver=_denominator_dim,
            carrier_role="denominator",
            anchor_role="numerator_companion",
            composition_roles=_composition_roles_for_mode("numerator_companion"),
        )
    raise ValueError(f"unsupported fractional variant: {variant}")


def _make_affine_spec() -> OperatorSpec:
    return OperatorSpec(
        family="affine",
        operator_kind="affine_latent",
        operator_id="affine:latent",
        scaffold_id_builder=_static_scaffold_id("affine:latent"),
        parent_template=("const", 0.0),
        hole_path=(),
        target_mode="robust",
        form="affine_latent",
        composition_mode="latent",
        carrier_slot=None,
        carrier_dim_resolver=_target_dim_resolver,
        subset_slot="terms",
        subset_max_arity=3,
        subset_role="affine_term",
        composition_roles=_composition_roles_for_mode("latent"),
    )


def _make_power_spec(
    *,
    kind: str,
    exponent: float,
    anchored: bool,
    template: tuple,
    hole_path: tuple[int, ...],
    form: str,
) -> OperatorSpec:
    kind_token = str(kind).strip().lower()
    if anchored:
        return OperatorSpec(
            family="power",
            operator_kind="power_wrap",
            operator_id=f"power:{kind_token}_mul",
            scaffold_id_builder=_anchored_scaffold_id(f"power:{kind_token}_mul"),
            parent_template=("mul", "__ANCHOR__", template),
            hole_path=(2, *tuple(hole_path)),
            target_mode="full",
            form=f"{form}_mul",
            composition_mode="prefactor",
            wrap_op=kind_token,
            exponent=float(exponent),
            anchor_mode="per_anchor",
            anchor_slot="anchor",
            carrier_dim_resolver=_power_mul_dim(float(exponent)),
            anchor_role="prefactor",
            composition_roles=_composition_roles_for_mode("prefactor"),
        )
    return OperatorSpec(
        family="power",
        operator_kind="power_wrap",
        operator_id=f"power:{kind_token}",
        scaffold_id_builder=_static_scaffold_id(f"power:{kind_token}"),
        parent_template=template,
        hole_path=hole_path,
        target_mode="full",
        form=form,
        composition_mode="base",
        wrap_op=kind_token,
        exponent=float(exponent),
        carrier_dim_resolver=_power_base_dim(float(exponent)),
        composition_roles=_composition_roles_for_mode("base"),
    )


def _make_quadratic_spec(*, variant: str) -> OperatorSpec:
    variant_token = str(variant).strip().lower()
    if variant_token == "sqrt":
        return _make_composed_wrap_spec(
            family="quadratic",
            operator_kind="quadratic_wrap",
            operator_id="quadratic:sqrt",
            scaffold_id_builder=_static_scaffold_id("quadratic:sqrt"),
            parent_template=("sqrt", "__CARRIER__"),
            hole_path=(1,),
            target_mode="robust",
            form="quadratic_sqrt",
            composition_mode="base",
            wrap_op="sqrt",
            carrier_dim_resolver=_quadratic_latent_dim,
            carrier_role="quadratic_latent",
        )
    if variant_token == "sqrt_mul":
        return _make_composed_wrap_spec(
            family="quadratic",
            operator_kind="quadratic_wrap",
            operator_id="quadratic:sqrt_mul",
            scaffold_id_builder=_anchored_scaffold_id("quadratic:sqrt_mul"),
            parent_template=("mul", "__ANCHOR__", ("sqrt", "__CARRIER__")),
            hole_path=(2, 1),
            target_mode="robust",
            form="quadratic_sqrt_mul",
            composition_mode="prefactor",
            wrap_op="sqrt",
            anchor_mode="per_anchor",
            anchor_slot="anchor",
            carrier_dim_resolver=_quadratic_latent_mul_dim,
            carrier_role="quadratic_latent",
            anchor_role="prefactor",
        )
    raise ValueError(f"unsupported quadratic variant: {variant}")


_POWER_OPERATOR_DEFS: tuple[tuple[str, float, tuple, tuple[int, ...], str], ...] = (
    ("invsqrt", -0.5, ("div", ("const", 1.0), ("sqrt", "__CARRIER__")), (2, 1), "power_invsqrt"),
    ("neg2", -2.0, ("div", ("const", 1.0), ("sqr", "__CARRIER__")), (2, 1), "power_neg2"),
    ("inv", -1.0, ("div", ("const", 1.0), "__CARRIER__"), (2,), "power_inv"),
    ("sqrt", 0.5, ("sqrt", "__CARRIER__"), (1,), "power_sqrt"),
    ("sqr", 2.0, ("sqr", "__CARRIER__"), (1,), "power_sqr"),
)

_OPERATOR_ALGEBRA_SPECS: tuple[OperatorSpec, ...] = (
    _make_affine_spec(),
    _make_harmonic_spec(periodic_kind="sin", variant="base"),
    _make_harmonic_spec(periodic_kind="cos", variant="base"),
    _make_harmonic_spec(periodic_kind="sin", variant="add"),
    _make_harmonic_spec(periodic_kind="cos", variant="add"),
    _make_harmonic_spec(periodic_kind="sin", variant="mul"),
    _make_harmonic_spec(periodic_kind="cos", variant="mul"),
    _make_unary_wrap_spec(family="exp", wrap_op="exp", variant="base"),
    _make_unary_wrap_spec(family="exp", wrap_op="exp", variant="add"),
    _make_unary_wrap_spec(family="exp", wrap_op="exp", variant="mul"),
    _make_unary_wrap_spec(family="log", wrap_op="log", variant="base"),
    _make_unary_wrap_spec(family="log", wrap_op="log", variant="add"),
    _make_fractional_spec(variant="affine"),
    _make_fractional_spec(variant="num_over_anchor"),
    _make_fractional_spec(variant="anchor_over_den"),
    *tuple(
        _make_power_spec(
            kind=kind,
            exponent=float(exponent),
            anchored=anchored,
            template=template,
            hole_path=hole_path,
            form=form,
        )
        for kind, exponent, template, hole_path, form in _POWER_OPERATOR_DEFS
        for anchored in (True, False)
    ),
    _make_quadratic_spec(variant="sqrt"),
    _make_quadratic_spec(variant="sqrt_mul"),
)

_OPERATOR_SPEC_BY_ID: dict[str, OperatorSpec] = {
    str(spec.operator_id): spec for spec in _OPERATOR_ALGEBRA_SPECS
}

_FAMILY_OPERATOR_PRESETS: dict[str, tuple[str, ...]] = {
    "affine": ("affine:latent",),
    "periodic": (
        "periodic:sin_base",
        "periodic:cos_base",
        "periodic:sin_add",
        "periodic:cos_add",
        "periodic:sin_mul",
        "periodic:cos_mul",
    ),
    "exp": ("exp:base", "exp:add", "exp:mul"),
    "log": ("log:base", "log:add"),
    "rational": ("rational:affine", "rational:num_over_anchor", "rational:anchor_over_den"),
    "power": (
        "power:invsqrt_mul",
        "power:invsqrt",
        "power:sqrt_mul",
        "power:sqrt",
        "power:sqr_mul",
        "power:sqr",
        "power:inv_mul",
        "power:inv",
        "power:neg2_mul",
        "power:neg2",
    ),
    "quadratic": ("quadratic:sqrt", "quadratic:sqrt_mul"),
}


def operator_algebra_specs() -> list[OperatorSpec]:
    return list(_OPERATOR_ALGEBRA_SPECS)


def family_operator_preset_ids(family: str) -> tuple[str, ...]:
    token = str(family or "").strip().lower()
    return tuple(_FAMILY_OPERATOR_PRESETS.get(token, ()))


def family_operator_specs(family: str) -> list[OperatorSpec]:
    wanted = family_operator_preset_ids(family)
    return [spec for operator_id in wanted if (spec := _OPERATOR_SPEC_BY_ID.get(str(operator_id))) is not None]


__all__ = [
    "OperatorSpec",
    "family_operator_preset_ids",
    "family_operator_specs",
    "operator_algebra_specs",
]
