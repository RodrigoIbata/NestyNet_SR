# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch

from ..closures import (
    BoundClosure,
    ClosureDesign,
    make_direct_affine_closure,
    make_direct_linear_wrap_closure,
    make_direct_periodic_closure,
    make_direct_power_closure,
    make_direct_quadratic_closure,
    make_direct_rational_closure,
    make_multi_term_rational_closure,
)
from ..expr_ast import simplify


@dataclass(frozen=True)
class BuiltClosureCandidate:
    bound_closure: BoundClosure
    design: ClosureDesign
    generation_source: str
    tuple_provenance: str
    proposal_family: str
    local_mapping_kind: str
    local_mapping_nparams: int | None = None
    direct_metadata: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class LinearWrapBuildSpec:
    family: str
    wrap_op: str
    anchor_role_by_kind: Mapping[str, str]
    local_mapping_by_kind: Mapping[str, str]
    carrier_dim_rule: Any = "dimless"
    carrier_domain_rule: Any = None


LINEAR_WRAP_BUILD_SPECS: dict[str, LinearWrapBuildSpec] = {
    "exp": LinearWrapBuildSpec(
        family="exp",
        wrap_op="exp",
        anchor_role_by_kind={"add": "companion", "mul": "envelope"},
        local_mapping_by_kind={
            "base": "direct_exp_base_head",
            "add": "direct_exp_add_head",
            "mul": "direct_exp_mul_head",
        },
    ),
    "log": LinearWrapBuildSpec(
        family="log",
        wrap_op="log",
        anchor_role_by_kind={"add": "companion", "mul": "companion"},
        local_mapping_by_kind={
            "base": "direct_log_base_head",
            "add": "direct_log_add_head",
        },
        carrier_domain_rule="positive_output",
    ),
}


def build_closure_candidate(
    *,
    bound_closure: BoundClosure,
    design: ClosureDesign,
    generation_source: str,
    tuple_provenance: str,
    proposal_family: str,
    local_mapping_kind: str,
    local_mapping_nparams: int | None = None,
    direct_metadata: Mapping[str, Any] | None = None,
) -> BuiltClosureCandidate:
    return build_candidate(
        bound_closure=bound_closure,
        design=design,
        generation_source=generation_source,
        tuple_provenance=tuple_provenance,
        proposal_family=proposal_family,
        local_mapping_kind=local_mapping_kind,
        local_mapping_nparams=local_mapping_nparams,
        direct_metadata=direct_metadata,
    )


def build_candidate(
    *,
    bound_closure: BoundClosure,
    design: ClosureDesign,
    generation_source: str,
    tuple_provenance: str,
    proposal_family: str,
    local_mapping_kind: str,
    local_mapping_nparams: int | None = None,
    direct_metadata: Mapping[str, Any] | None = None,
) -> BuiltClosureCandidate:
    return BuiltClosureCandidate(
        bound_closure=bound_closure,
        design=design,
        generation_source=str(generation_source),
        tuple_provenance=str(tuple_provenance),
        proposal_family=str(proposal_family),
        local_mapping_kind=str(local_mapping_kind),
        local_mapping_nparams=None if local_mapping_nparams is None else int(local_mapping_nparams),
        direct_metadata=dict(direct_metadata or {}),
    )


def build_linear_combo_candidate(
    *,
    bound_closure: BoundClosure,
    fit_matrix: torch.Tensor,
    probe_matrix: torch.Tensor,
    terms: Sequence[tuple],
    bias_index: int,
    design_metadata: Mapping[str, Any] | None,
    generation_source: str,
    tuple_provenance: str,
    proposal_family: str,
    local_mapping_kind: str,
    local_mapping_nparams: int | None = None,
    direct_metadata: Mapping[str, Any] | None = None,
) -> BuiltClosureCandidate:
    return build_candidate(
        bound_closure=bound_closure,
        design=ClosureDesign(
            fit_matrix=fit_matrix,
            probe_matrix=probe_matrix,
            materializer="linear_combo",
            materializer_payload={
                "terms": list(terms),
                "bias_index": int(bias_index),
            },
            metadata=dict(design_metadata or {}),
        ),
        generation_source=generation_source,
        tuple_provenance=tuple_provenance,
        proposal_family=proposal_family,
        local_mapping_kind=local_mapping_kind,
        local_mapping_nparams=local_mapping_nparams,
        direct_metadata=direct_metadata,
    )


def build_materialized_candidate(
    *,
    bound_closure: BoundClosure,
    payload: Mapping[str, Any] | None,
    materializer: str,
    materializer_payload: Mapping[str, Any] | None,
    design_metadata: Mapping[str, Any] | None,
    generation_source: str,
    tuple_provenance: str,
    proposal_family: str,
    local_mapping_kind: str,
    local_mapping_nparams: int | None = None,
    direct_metadata: Mapping[str, Any] | None = None,
    fit_matrix: torch.Tensor | None = None,
    probe_matrix: torch.Tensor | None = None,
) -> BuiltClosureCandidate:
    return build_candidate(
        bound_closure=bound_closure,
        design=ClosureDesign(
            fit_matrix=fit_matrix,
            probe_matrix=probe_matrix,
            payload=dict(payload or {}),
            materializer=str(materializer),
            materializer_payload=dict(materializer_payload or {}),
            metadata=dict(design_metadata or {}),
        ),
        generation_source=generation_source,
        tuple_provenance=tuple_provenance,
        proposal_family=proposal_family,
        local_mapping_kind=local_mapping_kind,
        local_mapping_nparams=local_mapping_nparams,
        direct_metadata=direct_metadata,
    )


def build_generic_linear_wrap_candidate(
    *,
    family: str,
    wrap_kind: str,
    wrap_op: str,
    scaffold_id: str,
    hole_node: tuple,
    feature_node: tuple,
    anchor_node: tuple | None,
    fit_matrix: torch.Tensor,
    probe_matrix: torch.Tensor,
    terms: Sequence[tuple],
    bias_index: int,
    design_metadata: Mapping[str, Any] | None,
    generation_source: str,
    tuple_provenance: str,
    proposal_family: str,
    local_mapping_kind: str,
    anchor_role: str | None = None,
    carrier_dim_rule: Any = "dimless",
    carrier_domain_rule: Any = None,
    direct_metadata: Mapping[str, Any] | None = None,
) -> BuiltClosureCandidate:
    return build_linear_combo_candidate(
        bound_closure=make_direct_linear_wrap_closure(
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
        ),
        fit_matrix=fit_matrix,
        probe_matrix=probe_matrix,
        terms=terms,
        bias_index=int(bias_index),
        design_metadata=design_metadata,
        generation_source=generation_source,
        tuple_provenance=tuple_provenance,
        proposal_family=proposal_family,
        local_mapping_kind=local_mapping_kind,
        direct_metadata=direct_metadata,
    )


def build_generic_structured_candidate(
    *,
    bound_closure: BoundClosure,
    payload: Mapping[str, Any] | None,
    materializer: str,
    materializer_payload: Mapping[str, Any] | None,
    design_metadata: Mapping[str, Any] | None,
    generation_source: str,
    tuple_provenance: str,
    proposal_family: str,
    local_mapping_kind: str,
    local_mapping_nparams: int | None = None,
    direct_metadata: Mapping[str, Any] | None = None,
    fit_matrix: torch.Tensor | None = None,
    probe_matrix: torch.Tensor | None = None,
) -> BuiltClosureCandidate:
    return build_materialized_candidate(
        bound_closure=bound_closure,
        payload=payload,
        materializer=materializer,
        materializer_payload=materializer_payload,
        design_metadata=design_metadata,
        generation_source=generation_source,
        tuple_provenance=tuple_provenance,
        proposal_family=proposal_family,
        local_mapping_kind=local_mapping_kind,
        local_mapping_nparams=local_mapping_nparams,
        direct_metadata=direct_metadata,
        fit_matrix=fit_matrix,
        probe_matrix=probe_matrix,
    )


def build_affine_latent_candidate(
    *,
    scaffold_id: str,
    term_nodes: Sequence[tuple],
    fit_matrix: torch.Tensor,
    probe_matrix: torch.Tensor,
    source: str,
) -> BuiltClosureCandidate:
    terms = [node for node in list(term_nodes or ()) if isinstance(node, tuple)]
    return build_linear_combo_candidate(
        bound_closure=make_direct_affine_closure(
            scaffold_id=str(scaffold_id),
            term_nodes=tuple(terms),
        ),
        fit_matrix=fit_matrix,
        probe_matrix=probe_matrix,
        terms=terms,
        bias_index=int(len(terms)),
        design_metadata={"source": str(source)},
        generation_source=f"closure_search_direct_affine:{source}",
        tuple_provenance="closure_search_direct_affine",
        proposal_family="closure_search_direct_affine",
        local_mapping_kind="direct_affine_head",
        local_mapping_nparams=int(len(terms) + 1),
        direct_metadata={
            "term_nodes": list(terms),
            "source": str(source),
        },
    )


def build_rational_affine_candidate(
    *,
    scaffold_id: str,
    u_node: tuple,
    v_node: tuple,
    u_fit: torch.Tensor,
    u_probe: torch.Tensor,
    v_fit: torch.Tensor,
    v_probe: torch.Tensor,
    max_depth: int,
    var_dims,
    y_dims,
    safe_eps: float,
    u_source: str,
    v_source: str,
) -> BuiltClosureCandidate:
    return build_generic_structured_candidate(
        bound_closure=make_direct_rational_closure(
            scaffold_id=str(scaffold_id),
            u_node=u_node,
            v_node=v_node,
        ),
        payload={
            "u_fit": u_fit,
            "u_probe": u_probe,
            "v_fit": v_fit,
            "v_probe": v_probe,
            "safe_eps": float(safe_eps),
        },
        materializer="rational_affine",
        materializer_payload={
            "u_node": u_node,
            "v_node": v_node,
            "max_depth": int(max_depth),
            "var_dims": var_dims,
            "y_dims": y_dims,
        },
        design_metadata={
            "u_source": str(u_source),
            "v_source": str(v_source),
        },
        generation_source=f"closure_search_direct_rational:{u_source}:{v_source}",
        tuple_provenance="closure_search_direct_rational",
        proposal_family="closure_search_direct_rational",
        local_mapping_kind="direct_rational_head",
        local_mapping_nparams=3,
        direct_metadata={
            "u_node": u_node,
            "v_node": v_node,
            "u_source": str(u_source),
            "v_source": str(v_source),
        },
    )


def build_multi_term_rational_candidate(
    *,
    scaffold_id: str,
    u_nodes: Sequence[tuple],
    v_nodes: Sequence[tuple],
    u_fits: Sequence[torch.Tensor],
    u_probes: Sequence[torch.Tensor],
    v_fits: Sequence[torch.Tensor],
    v_probes: Sequence[torch.Tensor],
    max_depth: int,
    var_dims,
    y_dims,
    safe_eps: float,
    u_sources: Sequence[str] | None = None,
    v_sources: Sequence[str] | None = None,
) -> BuiltClosureCandidate:
    u_src_list = [str(v) for v in list(u_sources or ["pool"] * len(list(u_nodes or [])))]
    v_src_list = [str(v) for v in list(v_sources or ["pool"] * len(list(v_nodes or [])))]
    return build_generic_structured_candidate(
        bound_closure=make_multi_term_rational_closure(
            scaffold_id=str(scaffold_id),
            u_nodes=tuple(u_nodes),
            v_nodes=tuple(v_nodes),
        ),
        payload={
            "u_fits": list(u_fits),
            "u_probes": list(u_probes),
            "v_fits": list(v_fits),
            "v_probes": list(v_probes),
            "safe_eps": float(safe_eps),
        },
        materializer="multi_term_rational",
        materializer_payload={
            "u_nodes": list(u_nodes),
            "v_nodes": list(v_nodes),
            "max_depth": int(max_depth),
            "var_dims": var_dims,
            "y_dims": y_dims,
        },
        design_metadata={
            "u_sources": u_src_list,
            "v_sources": v_src_list,
        },
        generation_source=f"closure_search_multi_term_rational:{'+'.join(u_src_list)}:{'+'.join(v_src_list)}",
        tuple_provenance="closure_search_multi_term_rational",
        proposal_family="closure_search_multi_term_rational",
        local_mapping_kind="multi_term_rational_head",
        local_mapping_nparams=int(len(list(u_nodes or [])) + 1 + len(list(v_nodes or []))),
        direct_metadata={
            "u_nodes": list(u_nodes),
            "v_nodes": list(v_nodes),
            "u_sources": u_src_list,
            "v_sources": v_src_list,
            "form": "multi_term_rational",
        },
    )


def build_affine_power_candidate(
    *,
    scaffold_id: str,
    power_kind: str,
    exponent: float,
    hole_node: tuple,
    anchor_node: tuple | None,
    h_fit: torch.Tensor,
    h_probe: torch.Tensor,
    anchor_fit: torch.Tensor | None,
    anchor_probe: torch.Tensor | None,
    max_depth: int,
    var_dims,
    y_dims,
    safe_eps: float,
    source: str,
    power_variant: str | None = None,
) -> BuiltClosureCandidate:
    variant_token = str(power_variant or "").strip().lower()
    return build_generic_structured_candidate(
        bound_closure=make_direct_power_closure(
            scaffold_id=str(scaffold_id),
            power_kind=str(power_kind),
            exponent=float(exponent),
            hole_node=hole_node,
            anchor_node=anchor_node,
        ),
        payload={
            "h_fit": h_fit,
            "h_probe": h_probe,
            "anchor_fit": anchor_fit,
            "anchor_probe": anchor_probe,
            "exponent": float(exponent),
            "power_variant": variant_token or None,
            "safe_eps": float(safe_eps),
        },
        materializer="affine_power",
        materializer_payload={
            "hole_node": hole_node,
            "anchor_node": anchor_node,
            "exponent": float(exponent),
            "power_variant": variant_token or None,
            "max_depth": int(max_depth),
            "var_dims": var_dims,
            "y_dims": y_dims,
        },
        design_metadata={
            "power_kind": str(power_kind),
            "power_variant": variant_token or "",
            "source": str(source),
        },
        generation_source=(
            f"closure_search_direct_power:{power_kind}:{variant_token or 'default'}:{source}"
        ),
        tuple_provenance="closure_search_direct_power",
        proposal_family="closure_search_direct_power",
        local_mapping_kind="direct_power_head",
        local_mapping_nparams=None,
        direct_metadata={
            "power_kind": str(power_kind),
            "power_exponent": float(exponent),
            "power_variant": variant_token or "",
            "hole_node": hole_node,
            "power_inner_node": hole_node,
            "anchor_node": anchor_node,
        },
    )


def build_quadratic_sqrt_candidate(
    *,
    scaffold_id: str,
    quadratic_kind: str,
    base_nodes: Sequence[tuple],
    anchor_node: tuple | None,
    quad_fit: torch.Tensor,
    quad_probe: torch.Tensor,
    anchor_fit: torch.Tensor | None,
    anchor_probe: torch.Tensor | None,
    max_depth: int,
    var_dims,
    y_dims,
    safe_eps: float,
    base_sources: Sequence[str],
) -> BuiltClosureCandidate:
    return build_generic_structured_candidate(
        bound_closure=make_direct_quadratic_closure(
            scaffold_id=str(scaffold_id),
            quadratic_kind=str(quadratic_kind),
            base_nodes=tuple(base_nodes),
            anchor_node=anchor_node,
        ),
        payload={
            "quad_fit": quad_fit,
            "quad_probe": quad_probe,
            "anchor_fit": anchor_fit,
            "anchor_probe": anchor_probe,
            "safe_eps": float(safe_eps),
        },
        materializer="quadratic_sqrt",
        materializer_payload={
            "base_nodes": list(base_nodes),
            "anchor_node": anchor_node,
            "max_depth": int(max_depth),
            "var_dims": var_dims,
            "y_dims": y_dims,
        },
        design_metadata={
            "quadratic_kind": str(quadratic_kind),
            "base_sources": [str(v) for v in list(base_sources or ())],
        },
        generation_source=f"closure_search_direct_quadratic:{quadratic_kind}:{'+'.join(str(v) for v in list(base_sources or ()))}",
        tuple_provenance="closure_search_direct_quadratic",
        proposal_family="closure_search_direct_quadratic",
        local_mapping_kind="direct_quadratic_sqrt_head",
        direct_metadata={
            "quadratic_kind": str(quadratic_kind),
            "quadratic_base_nodes": list(base_nodes),
            "anchor_node": anchor_node,
        },
    )


def build_linear_wrap_candidate(
    *,
    family: str,
    scaffold_id: str,
    kind: str,
    hole_node: tuple,
    feature_node: tuple,
    anchor_node: tuple | None,
    fit_matrix: torch.Tensor,
    probe_matrix: torch.Tensor,
    terms: Sequence[tuple],
    bias_index: int,
    source: str,
) -> BuiltClosureCandidate:
    family_token = str(family or "").strip().lower()
    kind_token = str(kind or "").strip().lower() or "base"
    spec = LINEAR_WRAP_BUILD_SPECS.get(family_token)
    if spec is None:
        raise ValueError(f"unsupported unary linear family: {family_token}")
    return build_generic_linear_wrap_candidate(
        family=family_token,
        wrap_kind=kind_token,
        wrap_op=str(spec.wrap_op),
        scaffold_id=str(scaffold_id),
        hole_node=hole_node,
        feature_node=feature_node,
        anchor_node=anchor_node,
        fit_matrix=fit_matrix,
        probe_matrix=probe_matrix,
        terms=terms,
        bias_index=int(bias_index),
        design_metadata={
            f"{family_token}_kind": kind_token,
            "source": str(source),
            "wrap_op": str(spec.wrap_op),
        },
        generation_source=f"closure_search_direct_{family_token}:{kind_token}:{source}",
        tuple_provenance=f"closure_search_direct_{family_token}",
        proposal_family=f"closure_search_direct_{family_token}",
        local_mapping_kind=spec.local_mapping_by_kind.get(kind_token, "direct_linear_head"),
        anchor_role=spec.anchor_role_by_kind.get(kind_token, None),
        carrier_dim_rule=spec.carrier_dim_rule,
        carrier_domain_rule=spec.carrier_domain_rule,
        direct_metadata={
            f"{family_token}_kind": kind_token,
            "hole_node": hole_node,
            "feature_node": feature_node,
            "wrap_op": str(spec.wrap_op),
        },
    )


def build_unary_linear_candidate(**kwargs) -> BuiltClosureCandidate:
    return build_linear_wrap_candidate(**kwargs)


def build_harmonic_periodic_candidate(
    *,
    scaffold_id: str,
    periodic_kind: str,
    hole_node: tuple,
    trig_node: tuple,
    cos_node: tuple,
    sin_node: tuple,
    anchor_node: tuple | None,
    envelope_node: tuple,
    companion_nodes: Sequence[tuple],
    fit_cols: Sequence[torch.Tensor],
    probe_cols: Sequence[torch.Tensor],
    source: str,
    mode: str,
    envelope_source: str,
    companion_sources: Sequence[str],
) -> BuiltClosureCandidate:
    env_cos_node = simplify(("mul", envelope_node, cos_node))
    env_sin_node = simplify(("mul", envelope_node, sin_node))
    return build_linear_combo_candidate(
        bound_closure=make_direct_periodic_closure(
            scaffold_id=str(scaffold_id),
            periodic_kind=str(periodic_kind),
            hole_node=hole_node,
            feature_node=trig_node,
            anchor_node=anchor_node,
            envelope_node=envelope_node,
            companion_nodes=tuple(companion_nodes),
            harmonic_feature_nodes=(cos_node, sin_node),
        ),
        fit_matrix=torch.stack(list(fit_cols), dim=1),
        probe_matrix=torch.stack(list(probe_cols), dim=1),
        terms=[env_cos_node, env_sin_node, *list(companion_nodes or ())],
        bias_index=int(len(list(fit_cols)) - 1),
        design_metadata={
            "periodic_kind": str(periodic_kind),
            "source": str(source),
            "mode": str(mode),
            "envelope_source": str(envelope_source),
        },
        generation_source=f"closure_search_direct_harmonic:{source}:{envelope_source}",
        tuple_provenance="closure_search_direct_periodic",
        proposal_family="closure_search_direct_periodic",
        local_mapping_kind="direct_harmonic_head",
        direct_metadata={
            "feature_kind": str(periodic_kind),
            "hole_node": hole_node,
            "feature_node": trig_node,
            "harmonic_feature_nodes": [cos_node, sin_node],
            "envelope_node": envelope_node,
            "companion_nodes": list(companion_nodes or ()),
            "envelope_source": str(envelope_source),
            "companion_sources": [str(v) for v in list(companion_sources or ())],
            "periodic_mode": str(mode),
        },
    )


def build_literal_periodic_candidate(
    *,
    scaffold_id: str,
    periodic_kind: str,
    hole_node: tuple,
    trig_node: tuple,
    anchor_node: tuple,
    child_expr: tuple,
    fit_matrix: torch.Tensor,
    probe_matrix: torch.Tensor,
    source: str,
) -> BuiltClosureCandidate:
    return build_materialized_candidate(
        bound_closure=make_direct_periodic_closure(
            scaffold_id=str(scaffold_id),
            periodic_kind=str(periodic_kind),
            hole_node=hole_node,
            feature_node=trig_node,
            anchor_node=anchor_node,
            companion_nodes=(anchor_node,),
            harmonic_feature_nodes=(trig_node,),
            expr=child_expr,
        ),
        payload=None,
        materializer="literal",
        materializer_payload={"expr": child_expr},
        design_metadata={"periodic_kind": str(periodic_kind), "source": str(source)},
        generation_source=f"closure_search_direct_{source}",
        tuple_provenance="closure_search_direct_periodic",
        proposal_family="closure_search_direct_periodic",
        local_mapping_kind="direct_linear_head",
        local_mapping_nparams=3,
        direct_metadata={
            "feature_kind": str(periodic_kind),
            "hole_node": hole_node,
            "feature_node": trig_node,
        },
        fit_matrix=fit_matrix,
        probe_matrix=probe_matrix,
    )


__all__ = [
    "BuiltClosureCandidate",
    "LinearWrapBuildSpec",
    "LINEAR_WRAP_BUILD_SPECS",
    "build_affine_power_candidate",
    "build_affine_latent_candidate",
    "build_candidate",
    "build_closure_candidate",
    "build_generic_linear_wrap_candidate",
    "build_generic_structured_candidate",
    "build_harmonic_periodic_candidate",
    "build_linear_combo_candidate",
    "build_linear_wrap_candidate",
    "build_literal_periodic_candidate",
    "build_materialized_candidate",
    "build_multi_term_rational_candidate",
    "build_quadratic_sqrt_candidate",
    "build_rational_affine_candidate",
    "build_unary_linear_candidate",
]
