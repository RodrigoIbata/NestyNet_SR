# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

import torch

from nestynet_sr.sr_search.factorized_search.atom_policy import (
    build_aux_policy_plan,
    build_atom_library,
    seed_blocks_from_atom_relations,
)
from nestynet_sr.sr_search.factorized_search.basis_scoring import materialize_direct_linear_combo
from nestynet_sr.sr_search.factorized_search.basis_state import ProposalContext
from nestynet_sr.sr_search.factorized_search.emergent_atoms import (
    EmergentAtom,
    harvest_emergent_atoms,
    merge_emergent_atom_registry,
    seed_blocks_from_emergent_atoms,
)
from nestynet_sr.sr_search.factorized_search.engine.proposal_execution import (
    _propose_atomized_linear_span_rows,
)
from nestynet_sr.sr_search.factorized_search.expr_ast import is_valid_node, node_str, simplify
from nestynet_sr.sr_search.factorized_search.proposal_families.runner import (
    run_closure_search_pass_impl,
)
from nestynet_sr.sr_search.factorized_search.proposal_families.scaffold_enum import (
    enumerate_operator_applications,
)
from nestynet_sr.sr_search.factorized_search.proposal_families.seed_blocks import (
    build_recursive_seed_pool,
)
from nestynet_sr.sr_search.factorized_search.proposal_families.steering import (
    allocate_family_budgets,
)
from nestynet_sr.sr_search.factorized_search.proposal_families.types import OperatorApplication


TARGET_DIM = (0.0, -3.0, 1.0, 0.0, 0.0)
DIMLESS = (0.0, 0.0, 0.0, 0.0, 0.0)


def _pb037_data(n=900):
    g = torch.Generator().manual_seed(37)
    x0 = 1.0 + 4.0 * torch.rand((n, 1), generator=g, dtype=torch.float64)
    x1 = 1.0 + 4.0 * torch.rand((n, 1), generator=g, dtype=torch.float64)
    x2 = 1.0 + 4.0 * torch.rand((n, 1), generator=g, dtype=torch.float64)
    x = torch.cat([x0, x1, x2], dim=1)
    y = x0 + x1 + 2.0 * torch.sqrt(x0 * x1) * torch.cos(x2)
    return x[: n // 2], y[: n // 2], x[n // 2 :], y[n // 2 :]


def _expr(node):
    return str(node_str(simplify(node)))


def _atom_exprs(atoms):
    return {str(node_str(atom.node)): atom for atom in atoms}


def _harvest(rows, *, var_dims=(TARGET_DIM, TARGET_DIM, DIMLESS), y_dims=TARGET_DIM, max_new=8):
    x_fit, y_fit, x_probe, y_probe = _pb037_data()
    stats = {}
    atoms = harvest_emergent_atoms(
        candidate_rows=rows,
        x_fit=x_fit,
        y_fit=y_fit,
        x_probe=x_probe,
        y_probe=y_probe,
        var_dims=var_dims,
        y_dims=y_dims,
        stats=stats,
        max_new=max_new,
        debug_limit=16,
    )
    return atoms, stats


def test_harvest_defactors_common_dimensionless_denominator():
    expr = (
        "add",
        ("div", ("var", 0), ("var", 2)),
        ("div", ("var", 1), ("var", 2)),
    )

    atoms, stats = _harvest(
        [
            {
                "expr": expr,
                "proposal_key": "shared_den",
                "scaffold_family": "rational",
                "local_fit_mse": 1.0,
                "local_probe_mse": 1.0,
            }
        ]
    )

    exprs = _atom_exprs(atoms)
    assert _expr(("add", ("var", 0), ("var", 1))) in exprs
    atom = exprs[_expr(("add", ("var", 0), ("var", 1)))]
    assert atom.kind == "target_term"
    assert atom.evidence["common_denominator_stripped"] is True
    assert stats["emergent_aux_atom_accepted"] >= 1


def test_harvest_rejects_mixed_denominator():
    expr = (
        "add",
        ("div", ("var", 0), ("var", 2)),
        ("div", ("var", 1), ("var", 0)),
    )

    atoms, _stats = _harvest(
        [
            {
                "expr": expr,
                "proposal_key": "mixed_den",
                "scaffold_family": "rational",
                "local_fit_mse": 1.0,
                "local_probe_mse": 1.0,
            }
        ]
    )

    assert _expr(("add", ("var", 0), ("var", 1))) not in _atom_exprs(atoms)


def test_harvest_rejects_dimensional_denominator_stripping():
    expr = (
        "add",
        ("div", ("var", 0), ("var", 2)),
        ("div", ("var", 1), ("var", 2)),
    )

    atoms, _stats = _harvest(
        [
            {
                "expr": expr,
                "proposal_key": "dimensional_den",
                "scaffold_family": "rational",
                "local_fit_mse": 1.0,
                "local_probe_mse": 1.0,
            }
        ],
        var_dims=(TARGET_DIM, TARGET_DIM, TARGET_DIM),
    )

    assert _expr(("add", ("var", 0), ("var", 1))) not in _atom_exprs(atoms)


def test_harvest_pb037_motifs():
    expr = (
        "add",
        ("add", ("var", 0), ("var", 1)),
        (
            "mul",
            ("sqrt", ("mul", ("var", 0), ("var", 1))),
            ("cos", ("var", 2)),
        ),
    )

    atoms, _stats = _harvest(
        [
            {
                "expr": expr,
                "proposal_key": "pb037_near_miss",
                "scaffold_family": "periodic",
                "local_fit_mse": 0.5,
                "local_probe_mse": 0.5,
            }
        ],
        max_new=8,
    )

    exprs = _atom_exprs(atoms)
    assert _expr(("sqrt", ("mul", ("var", 0), ("var", 1)))) in exprs
    assert _expr(("cos", ("var", 2))) in exprs
    assert _expr(("add", ("var", 0), ("var", 1))) in exprs
    sqrt_atom = exprs[_expr(("sqrt", ("mul", ("var", 0), ("var", 1))))]
    cos_atom = exprs[_expr(("cos", ("var", 2)))]
    assert sqrt_atom.evidence["active_vars"] == [0, 1]
    assert sqrt_atom.evidence["active_dimensionless_vars"] == []
    assert sqrt_atom.evidence["active_dimensional_vars"] == [0, 1]
    assert cos_atom.evidence["active_vars"] == [2]
    assert cos_atom.evidence["active_dimensionless_vars"] == [2]


def test_harvest_strips_numeric_multipliers_from_aux_atoms():
    atoms, _stats = _harvest(
        [
            {
                "expr": ("mul", ("const", 0.273387), ("sin", ("var", 2))),
                "proposal_key": "scaled_sine",
                "scaffold_family": "rational",
                "local_fit_mse": 0.5,
                "local_probe_mse": 0.5,
            }
        ],
        max_new=4,
    )

    exprs = set(_atom_exprs(atoms))
    assert _expr(("sin", ("var", 2))) in exprs
    assert not any("0.273387" in expr for expr in exprs)


def test_harvest_keeps_dimensional_atom_when_dimensionless_repeats():
    rows = [
        {
            "expr": ("sin", ("var", 2)),
            "proposal_key": f"dimless_{idx}",
            "scaffold_family": "rational",
            "local_fit_mse": 0.1,
            "local_probe_mse": 0.1,
        }
        for idx in range(20)
    ]
    rows.append(
        {
            "expr": (
                "mul",
                ("add", ("var", 0), ("var", 1)),
                ("cos", ("var", 2)),
            ),
            "proposal_key": "target_sum",
            "scaffold_family": "rational",
            "local_fit_mse": 0.2,
            "local_probe_mse": 0.2,
        }
    )

    atoms, _stats = _harvest(rows, max_new=2)
    exprs = set(_atom_exprs(atoms))
    assert _expr(("add", ("var", 0), ("var", 1))) in exprs
    assert any(expr in exprs for expr in {_expr(("sin", ("var", 2))), _expr(("cos", ("var", 2)))})


def test_harvest_three_atoms_can_keep_pb037_target_carrier_and_trig():
    rows = [
        {
            "expr": (
                "mul",
                ("add", ("var", 0), ("var", 1)),
                ("cos", ("var", 2)),
            ),
            "proposal_key": "sum_times_cos",
            "scaffold_family": "rational",
            "local_fit_mse": 0.2,
            "local_probe_mse": 0.2,
        },
        {
            "expr": (
                "mul",
                ("const", 1.13918),
                ("sqrt", ("mul", ("var", 0), ("var", 1))),
            ),
            "proposal_key": "scaled_sqrt_product",
            "scaffold_family": "rational",
            "local_fit_mse": 0.3,
            "local_probe_mse": 0.3,
        },
    ]

    atoms, _stats = _harvest(rows, max_new=3)
    exprs = set(_atom_exprs(atoms))
    assert _expr(("add", ("var", 0), ("var", 1))) in exprs
    assert _expr(("sqrt", ("mul", ("var", 0), ("var", 1)))) in exprs
    assert _expr(("cos", ("var", 2))) in exprs


def test_harvest_role_buckets_keep_completed_modulator_over_dimless_artifact():
    artifact_rows = [
        {
            "expr": ("mul", ("sqrt", ("var", 2)), ("sqrt", ("var", 2))),
            "proposal_key": f"dimless_artifact_{idx}",
            "scaffold_family": "rational",
            "local_fit_mse": 0.01,
            "local_probe_mse": 0.01,
        }
        for idx in range(12)
    ]
    motif_row = {
        "expr": (
            "add",
            ("add", ("var", 0), ("var", 1)),
            (
                "mul",
                ("sqrt", ("mul", ("var", 0), ("var", 1))),
                ("cos", ("var", 2)),
            ),
        ),
        "proposal_key": "pb037_near_miss",
        "scaffold_family": "periodic",
        "local_fit_mse": 0.2,
        "local_probe_mse": 0.2,
    }

    atoms, stats = _harvest(artifact_rows + [motif_row], max_new=3)
    exprs = set(_atom_exprs(atoms))
    product_expr = _expr(
        (
            "mul",
            ("sqrt", ("mul", ("var", 0), ("var", 1))),
            ("cos", ("var", 2)),
        )
    )
    assert _expr(("add", ("var", 0), ("var", 1))) in exprs
    assert _expr(("sqrt", ("mul", ("var", 0), ("var", 1)))) in exprs or product_expr in exprs
    assert _expr(("cos", ("var", 2))) in exprs
    assert stats["emergent_aux_atom_observed_bucket_counts"]["completed_modulator"] >= 1


def test_harvest_observation_sink_keeps_seen_but_not_selected_atoms_for_policy():
    x_fit, y_fit, x_probe, y_probe = _pb037_data()
    observed = []
    stats = {}
    rows = [
        {
            "expr": (
                "add",
                ("add", ("var", 0), ("var", 1)),
                (
                    "mul",
                    ("sqrt", ("mul", ("var", 0), ("var", 1))),
                    ("cos", ("var", 2)),
                ),
            ),
            "proposal_key": "pb037_near_miss",
            "scaffold_family": "periodic",
            "local_fit_mse": 0.2,
            "local_probe_mse": 0.2,
        }
    ]

    selected = harvest_emergent_atoms(
        candidate_rows=rows,
        x_fit=x_fit,
        y_fit=y_fit,
        x_probe=x_probe,
        y_probe=y_probe,
        var_dims=(TARGET_DIM, TARGET_DIM, DIMLESS),
        y_dims=TARGET_DIM,
        stats=stats,
        max_new=2,
        debug_limit=8,
        observed_atom_sink=observed,
    )
    selected_exprs = set(_atom_exprs(selected))
    observed_exprs = set(_atom_exprs(observed))
    assert len(observed_exprs) > len(selected_exprs)
    assert stats["emergent_aux_atom_seen_not_retained"]

    library = build_atom_library(
        tuple(selected) + tuple(observed),
        var_dims=(TARGET_DIM, TARGET_DIM, DIMLESS),
        y_dims=TARGET_DIM,
    )
    product_expr = _expr(
        (
            "mul",
            ("sqrt", ("mul", ("var", 0), ("var", 1))),
            ("cos", ("var", 2)),
        )
    )
    assert product_expr in {str(node_str(relation.node)) for relation in library.relations}


def test_registry_rational_cap_does_not_let_dimless_atoms_starve_carriers():
    atoms = (
        EmergentAtom(
            node=simplify(("sin", ("var", 2))),
            dim=DIMLESS,
            kind="dimensionless_feature",
            score=100.0,
            evidence={"rational_derived": True, "sources": ["unit"]},
            source_count=1,
        ),
        EmergentAtom(
            node=simplify(("cos", ("var", 2))),
            dim=DIMLESS,
            kind="dimensionless_feature",
            score=90.0,
            evidence={"rational_derived": True, "sources": ["unit"]},
            source_count=1,
        ),
        EmergentAtom(
            node=simplify(("add", ("var", 0), ("var", 1))),
            dim=TARGET_DIM,
            kind="target_term",
            score=10.0,
            evidence={"rational_derived": True, "sources": ["unit"]},
            source_count=1,
        ),
        EmergentAtom(
            node=simplify(("sqrt", ("mul", ("var", 0), ("var", 1)))),
            dim=TARGET_DIM,
            kind="carrier",
            score=9.0,
            evidence={"rational_derived": True, "sources": ["unit"]},
            source_count=1,
        ),
    )

    registry = merge_emergent_atom_registry(
        (),
        atoms,
        max_total=4,
        max_target=2,
        max_dimensionless=2,
        max_rational_derived=2,
    )
    exprs = set(_atom_exprs(registry))
    assert _expr(("add", ("var", 0), ("var", 1))) in exprs
    assert _expr(("sqrt", ("mul", ("var", 0), ("var", 1)))) in exprs
    assert _expr(("sin", ("var", 2))) in exprs
    assert _expr(("cos", ("var", 2))) in exprs


def test_registry_reserves_slot_for_pure_dimensional_carrier():
    mixed_carrier = simplify(
        (
            "mul",
            ("add", ("var", 0), ("var", 1)),
            ("cos", ("var", 2)),
        )
    )
    pure_carrier = simplify(("sqrt", ("mul", ("var", 0), ("var", 1))))
    atoms = (
        EmergentAtom(
            node=mixed_carrier,
            dim=TARGET_DIM,
            kind="carrier",
            score=100.0,
            evidence={
                "sources": ["unit"],
                "active_vars": [0, 1, 2],
                "active_dimensionless_vars": [2],
                "active_dimensional_vars": [0, 1],
                "root_op": "mul",
            },
            source_count=1,
        ),
        EmergentAtom(
            node=simplify(("add", ("var", 0), ("var", 1))),
            dim=TARGET_DIM,
            kind="target_term",
            score=30.0,
            evidence={
                "sources": ["unit"],
                "active_vars": [0, 1],
                "active_dimensionless_vars": [],
                "active_dimensional_vars": [0, 1],
                "root_op": "add",
            },
            source_count=1,
        ),
        EmergentAtom(
            node=simplify(("cos", ("var", 2))),
            dim=DIMLESS,
            kind="dimensionless_feature",
            score=20.0,
            evidence={
                "sources": ["unit"],
                "active_vars": [2],
                "active_dimensionless_vars": [2],
                "active_dimensional_vars": [],
                "root_op": "cos",
            },
            source_count=1,
        ),
        EmergentAtom(
            node=pure_carrier,
            dim=TARGET_DIM,
            kind="carrier",
            score=1.0,
            evidence={
                "sources": ["unit"],
                "active_vars": [0, 1],
                "active_dimensionless_vars": [],
                "active_dimensional_vars": [0, 1],
                "root_op": "sqrt",
            },
            source_count=1,
        ),
    )

    stats = {}
    registry = merge_emergent_atom_registry(
        (),
        atoms,
        max_total=3,
        max_target=2,
        max_dimensionless=1,
        max_rational_derived=4,
        stats=stats,
    )
    exprs = set(_atom_exprs(registry))
    assert _expr(("add", ("var", 0), ("var", 1))) in exprs
    assert _expr(("cos", ("var", 2))) in exprs
    assert _expr(("sqrt", ("mul", ("var", 0), ("var", 1)))) in exprs
    assert _expr(mixed_carrier) not in exprs
    assert stats["emergent_aux_atom_registry_by_bucket"]["pure_dimensional_carrier"] == 1


def test_proposal_context_serializes_aux_blocks():
    atom = EmergentAtom(
        node=simplify(("add", ("var", 0), ("var", 1))),
        dim=TARGET_DIM,
        kind="target_term",
        score=1.0,
        evidence={"sources": ["unit"]},
        source_count=1,
    )
    blocks = seed_blocks_from_emergent_atoms((atom,), var_dims=(TARGET_DIM, TARGET_DIM, DIMLESS))
    payload = ProposalContext(aux_seed_blocks=blocks).to_dict()

    assert payload["aux_seed_blocks"][0]["expr"] == _expr(("add", ("var", 0), ("var", 1)))
    assert payload["aux_seed_blocks"][0]["metadata"]["origin"] == "aux:emergent"


def test_runner_augmented_lane_activates_with_aux_atoms():
    atom = EmergentAtom(
        node=simplify(("add", ("var", 0), ("var", 1))),
        dim=TARGET_DIM,
        kind="target_term",
        score=1.0,
        evidence={"sources": ["unit"]},
        source_count=1,
    )
    aux_blocks = seed_blocks_from_emergent_atoms((atom,), var_dims=(TARGET_DIM, TARGET_DIM, DIMLESS))
    calls = []

    def fake_enum(**kwargs):
        mode = str(kwargs.get("basis_seed_mode", ""))
        calls.append((mode, len(tuple(kwargs.get("aux_seed_blocks", ()) or ()))))
        count = int(kwargs.get("max_scaffolds", 0) or 0)
        return [
            OperatorApplication(
                family="affine",
                operator_id=f"{mode}:{idx}",
                scaffold_id=f"{mode}:{idx}",
                parent_node=("var", 0),
                hole_path=(),
            )
            for idx in range(count)
        ]

    def fake_direct(_spec, **_kwargs):
        return [], "direct_not_supported", {}

    x_fit, y_fit, x_probe, y_probe = _pb037_data(40)
    ret = run_closure_search_pass_impl(
        families=("affine",),
        nvars=3,
        max_scaffolds=2,
        anchors_per_family=2,
        max_depth=4,
        poly_degree=2,
        x_fit=x_fit,
        y_fit=y_fit,
        x_probe=x_probe,
        y_probe=y_probe,
        var_dims=(TARGET_DIM, TARGET_DIM, DIMLESS),
        y_dims=TARGET_DIM,
        pool_nodes=(),
        pool_phi_fit=torch.empty((x_fit.shape[0], 0), dtype=torch.float64),
        pool_phi_probe=torch.empty((x_probe.shape[0], 0), dtype=torch.float64),
        pool_dims=(),
        safe_eps=1.0e-12,
        preview_topk=1,
        beam_cfg={},
        solver_kwargs={},
        proposal_context=ProposalContext(aux_seed_blocks=aux_blocks),
        enumerate_operator_applications_fn=fake_enum,
        solve_direct_operator_preview_rows_fn=fake_direct,
    )

    assert ("basis_augmented", len(aux_blocks)) in calls
    assert ret["stats"]["proposal_lane_budgets"]["basis_augmented"] > 0
    assert ret["stats"]["scaffolds_enumerated_by_lane"]["basis_augmented"] > 0


def test_runner_core_lane_family_allocator_strips_aux_policy_context():
    atom = EmergentAtom(
        node=simplify(("cos", ("var", 2))),
        dim=DIMLESS,
        kind="dimensionless_feature",
        score=1.0,
        evidence={"sources": ["unit"]},
        source_count=1,
    )
    aux_blocks = seed_blocks_from_emergent_atoms((atom,), var_dims=(TARGET_DIM, TARGET_DIM, DIMLESS))
    allocator_calls = []

    def fake_allocator(*, families, max_scaffolds, anchors_per_family, context=None):
        allocator_calls.append(
            {
                "max_scaffolds": int(max_scaffolds),
                "has_atom_library": getattr(context, "atom_library", None) is not None,
                "aux_seed_blocks": len(tuple(getattr(context, "aux_seed_blocks", ()) or ())),
            }
        )
        fams = [str(family) for family in tuple(families or ()) if str(family)]
        if not fams:
            return {"steered": False, "scores": {}, "entries": []}
        remaining = int(max_scaffolds)
        entries = []
        for idx, family in enumerate(fams):
            if remaining <= 0:
                break
            budget = 1 if idx + 1 < len(fams) else remaining
            remaining -= budget
            entries.append(
                {
                    "family": family,
                    "max_scaffolds": budget,
                    "anchors_per_family": int(anchors_per_family),
                    "priority_score": 1.0 if getattr(context, "atom_library", None) is not None else 0.0,
                    "reason": "test",
                }
            )
        return {"steered": True, "scores": {}, "score_decomposition": {}, "entries": entries}

    def fake_enum(**kwargs):
        mode = str(kwargs.get("basis_seed_mode", ""))
        count = int(kwargs.get("max_scaffolds", 0) or 0)
        family = str(tuple(kwargs.get("families", ("affine",)) or ("affine",))[0])
        return [
            OperatorApplication(
                family=family,
                operator_id=f"{mode}:{family}:{idx}",
                scaffold_id=f"{mode}:{family}:{idx}",
                parent_node=("var", 0),
                hole_path=(),
            )
            for idx in range(count)
        ]

    def fake_direct(_spec, **_kwargs):
        return [], "direct_not_supported", {}

    x_fit, y_fit, x_probe, y_probe = _pb037_data(40)
    ret = run_closure_search_pass_impl(
        families=("periodic", "affine"),
        nvars=3,
        max_scaffolds=8,
        anchors_per_family=2,
        max_depth=4,
        poly_degree=2,
        x_fit=x_fit,
        y_fit=y_fit,
        x_probe=x_probe,
        y_probe=y_probe,
        var_dims=(TARGET_DIM, TARGET_DIM, DIMLESS),
        y_dims=TARGET_DIM,
        pool_nodes=(),
        pool_phi_fit=torch.empty((x_fit.shape[0], 0), dtype=torch.float64),
        pool_phi_probe=torch.empty((x_probe.shape[0], 0), dtype=torch.float64),
        pool_dims=(),
        safe_eps=1.0e-12,
        preview_topk=1,
        beam_cfg={},
        solver_kwargs={},
        proposal_context=ProposalContext(aux_seed_blocks=aux_blocks, atom_library=object()),
        family_allocator_fn=fake_allocator,
        enumerate_operator_applications_fn=fake_enum,
        solve_direct_operator_preview_rows_fn=fake_direct,
    )

    core_budget = int(ret["stats"]["proposal_lane_budgets"]["core"])
    core_calls = [call for call in allocator_calls if call["max_scaffolds"] == core_budget]
    assert core_calls
    assert all(not call["has_atom_library"] and call["aux_seed_blocks"] == 0 for call in core_calls)
    assert any(call["has_atom_library"] for call in allocator_calls)


def test_runner_protects_aux_affine_scaffolds_when_families_are_steered_elsewhere():
    atoms = (
        EmergentAtom(
            node=simplify(("add", ("var", 0), ("var", 1))),
            dim=TARGET_DIM,
            kind="target_term",
            score=3.0,
            evidence={"sources": ["unit"]},
            source_count=1,
        ),
        EmergentAtom(
            node=simplify(("cos", ("var", 2))),
            dim=DIMLESS,
            kind="dimensionless_feature",
            score=2.0,
            evidence={"sources": ["unit"]},
            source_count=1,
        ),
    )
    aux_blocks = seed_blocks_from_emergent_atoms(atoms, var_dims=(TARGET_DIM, TARGET_DIM, DIMLESS))
    seen_specs = []

    def fake_direct(spec, **_kwargs):
        seen_specs.append(spec)
        return [], "direct_not_supported", {}

    x_fit, y_fit, x_probe, y_probe = _pb037_data(40)
    ret = run_closure_search_pass_impl(
        families=("rational",),
        nvars=3,
        max_scaffolds=4,
        anchors_per_family=4,
        max_depth=4,
        poly_degree=2,
        x_fit=x_fit,
        y_fit=y_fit,
        x_probe=x_probe,
        y_probe=y_probe,
        var_dims=(TARGET_DIM, TARGET_DIM, DIMLESS),
        y_dims=TARGET_DIM,
        pool_nodes=(),
        pool_phi_fit=torch.empty((x_fit.shape[0], 0), dtype=torch.float64),
        pool_phi_probe=torch.empty((x_probe.shape[0], 0), dtype=torch.float64),
        pool_dims=(),
        safe_eps=1.0e-12,
        preview_topk=1,
        beam_cfg={},
        solver_kwargs={},
        proposal_context=ProposalContext(aux_seed_blocks=aux_blocks),
        solve_direct_operator_preview_rows_fn=fake_direct,
    )

    binding_texts = [str(spec.metadata.get("slot_bindings", "")) for spec in seen_specs]
    assert any(spec.family == "affine" for spec in seen_specs)
    assert any(_expr(("add", ("var", 0), ("var", 1))) in text for text in binding_texts)
    assert ret["stats"]["protected_aux_scaffolds_enumerated"] > 0


def test_scaffold_enum_receives_aux_seed_blocks():
    atoms = (
        EmergentAtom(
            node=simplify(("add", ("var", 0), ("var", 1))),
            dim=TARGET_DIM,
            kind="target_term",
            score=1.0,
            evidence={"sources": ["unit"]},
            source_count=1,
        ),
    )
    aux_blocks = seed_blocks_from_emergent_atoms(atoms, var_dims=(TARGET_DIM, TARGET_DIM, DIMLESS))

    apps_aug = enumerate_operator_applications(
        families=("affine",),
        nvars=3,
        y_dims=TARGET_DIM,
        var_dims=(TARGET_DIM, TARGET_DIM, DIMLESS),
        pool_nodes=(),
        pool_dims=(),
        anchors_per_family=4,
        max_scaffolds=20,
        aux_seed_blocks=aux_blocks,
        basis_seed_mode="basis_augmented",
    )
    apps_core = enumerate_operator_applications(
        families=("affine",),
        nvars=3,
        y_dims=TARGET_DIM,
        var_dims=(TARGET_DIM, TARGET_DIM, DIMLESS),
        pool_nodes=(),
        pool_dims=(),
        anchors_per_family=4,
        max_scaffolds=20,
        aux_seed_blocks=aux_blocks,
        basis_seed_mode="core_only",
    )

    aux_expr = _expr(("add", ("var", 0), ("var", 1)))
    assert any(aux_expr in str(app.metadata.get("slot_bindings", "")) for app in apps_aug)
    assert not any(aux_expr in str(app.metadata.get("slot_bindings", "")) for app in apps_core)


def test_recursive_pool_builds_pb037_product_from_aux_atoms():
    atoms = (
        EmergentAtom(
            node=simplify(("sqrt", ("mul", ("var", 0), ("var", 1)))),
            dim=TARGET_DIM,
            kind="carrier",
            score=2.0,
            evidence={"sources": ["unit"]},
            source_count=1,
        ),
        EmergentAtom(
            node=simplify(("cos", ("var", 2))),
            dim=DIMLESS,
            kind="dimensionless_feature",
            score=2.0,
            evidence={"sources": ["unit"]},
            source_count=1,
        ),
    )
    aux_blocks = seed_blocks_from_emergent_atoms(atoms, var_dims=(TARGET_DIM, TARGET_DIM, DIMLESS))
    pool = build_recursive_seed_pool(
        aux_blocks,
        rounds=1,
        include_product=True,
        include_monomial=False,
        include_quadratic=False,
        include_affine=False,
        product_max_arity=2,
        product_limit=8,
        max_nonlinear_depth=2,
    )

    product = _expr(
        (
            "mul",
            ("sqrt", ("mul", ("var", 0), ("var", 1))),
            ("cos", ("var", 2)),
        )
    )
    assert product in {str(node_str(block.node)) for block in pool if is_valid_node(block.node)}


def test_affine_combos_can_include_sum_and_pb037_product():
    atoms = (
        EmergentAtom(
            node=simplify(("add", ("var", 0), ("var", 1))),
            dim=TARGET_DIM,
            kind="target_term",
            score=3.0,
            evidence={"sources": ["unit"]},
            source_count=1,
        ),
        EmergentAtom(
            node=simplify(("sqrt", ("mul", ("var", 0), ("var", 1)))),
            dim=TARGET_DIM,
            kind="carrier",
            score=2.0,
            evidence={"sources": ["unit"]},
            source_count=1,
        ),
        EmergentAtom(
            node=simplify(("cos", ("var", 2))),
            dim=DIMLESS,
            kind="dimensionless_feature",
            score=2.0,
            evidence={"sources": ["unit"]},
            source_count=1,
        ),
    )
    aux_blocks = seed_blocks_from_emergent_atoms(atoms, var_dims=(TARGET_DIM, TARGET_DIM, DIMLESS))
    apps = enumerate_operator_applications(
        families=("affine",),
        nvars=3,
        y_dims=TARGET_DIM,
        var_dims=(TARGET_DIM, TARGET_DIM, DIMLESS),
        pool_nodes=(),
        pool_dims=(),
        anchors_per_family=8,
        max_scaffolds=80,
        aux_seed_blocks=aux_blocks,
        basis_seed_mode="basis_augmented",
    )

    sum_expr = _expr(("add", ("var", 0), ("var", 1)))
    product_expr = _expr(
        (
            "mul",
            ("sqrt", ("mul", ("var", 0), ("var", 1))),
            ("cos", ("var", 2)),
        )
    )
    binding_texts = [str(app.metadata.get("slot_bindings", "")) for app in apps]
    assert any(sum_expr in text for text in binding_texts)
    assert any(product_expr in text for text in binding_texts)


def test_affine_combos_prioritize_aux_sum_plus_carrier_trig_product():
    atoms = (
        EmergentAtom(
            node=simplify(("add", ("var", 0), ("var", 1))),
            dim=TARGET_DIM,
            kind="target_term",
            score=3.0,
            evidence={"sources": ["unit"]},
            source_count=1,
        ),
        EmergentAtom(
            node=simplify(("sqrt", ("mul", ("var", 0), ("var", 1)))),
            dim=TARGET_DIM,
            kind="carrier",
            score=2.0,
            evidence={"sources": ["unit"]},
            source_count=1,
        ),
        EmergentAtom(
            node=simplify(("cos", ("var", 2))),
            dim=DIMLESS,
            kind="dimensionless_feature",
            score=2.0,
            evidence={"sources": ["unit"]},
            source_count=1,
        ),
    )
    aux_blocks = seed_blocks_from_emergent_atoms(atoms, var_dims=(TARGET_DIM, TARGET_DIM, DIMLESS))
    apps = enumerate_operator_applications(
        families=("affine",),
        nvars=3,
        y_dims=TARGET_DIM,
        var_dims=(TARGET_DIM, TARGET_DIM, DIMLESS),
        pool_nodes=(),
        pool_dims=(),
        anchors_per_family=8,
        max_scaffolds=4,
        aux_seed_blocks=aux_blocks,
        basis_seed_mode="basis_augmented",
    )

    sum_expr = _expr(("add", ("var", 0), ("var", 1)))
    product_expr = _expr(
        (
            "mul",
            ("sqrt", ("mul", ("var", 0), ("var", 1))),
            ("cos", ("var", 2)),
        )
    )
    binding_texts = [str(app.metadata.get("slot_bindings", "")) for app in apps]
    assert any(sum_expr in text and product_expr in text for text in binding_texts)


def test_atom_policy_family_budget_uses_library_with_decomposition():
    atoms = (
        EmergentAtom(
            node=simplify(("add", ("var", 0), ("var", 1))),
            dim=TARGET_DIM,
            kind="target_term",
            score=3.0,
            evidence={"sources": ["unit"], "best_probe_gain_rel": 0.2, "best_fit_gain_rel": 0.2},
            source_count=1,
            roles=("expr",),
            families=("affine",),
        ),
        EmergentAtom(
            node=simplify(("sqrt", ("mul", ("var", 0), ("var", 1)))),
            dim=TARGET_DIM,
            kind="carrier",
            score=2.0,
            evidence={"sources": ["unit"], "best_probe_gain_rel": 0.05, "best_fit_gain_rel": 0.05},
            source_count=1,
            roles=("head:numerator",),
            families=("rational",),
        ),
        EmergentAtom(
            node=simplify(("cos", ("var", 2))),
            dim=DIMLESS,
            kind="dimensionless_feature",
            score=2.0,
            evidence={"sources": ["unit"], "best_parent_probe": 0.4},
            source_count=1,
            roles=("atom",),
            families=("periodic",),
        ),
    )
    library = build_atom_library(atoms, var_dims=(TARGET_DIM, TARGET_DIM, DIMLESS), y_dims=TARGET_DIM)
    plan = allocate_family_budgets(
        families=("affine", "periodic", "rational"),
        max_scaffolds=9,
        anchors_per_family=4,
        context=ProposalContext(atom_library=library),
    )

    assert plan["steered"] is True
    assert plan["scores"]["affine"] > 0.0
    assert plan["scores"]["periodic"] > 0.0
    assert plan["score_decomposition"]["affine"]["atom_policy"] > 0.0
    assert plan["score_decomposition"]["periodic"]["top_atom_reasons"]


def test_atom_policy_relation_seed_blocks_include_target_dim_product():
    atoms = (
        EmergentAtom(
            node=simplify(("sqrt", ("mul", ("var", 0), ("var", 1)))),
            dim=TARGET_DIM,
            kind="carrier",
            score=2.0,
            evidence={"sources": ["unit"], "best_probe_gain_rel": 0.05},
            source_count=1,
            roles=("head:numerator",),
            families=("rational",),
        ),
        EmergentAtom(
            node=simplify(("cos", ("var", 2))),
            dim=DIMLESS,
            kind="dimensionless_feature",
            score=2.0,
            evidence={"sources": ["unit"], "best_parent_probe": 0.5},
            source_count=1,
            roles=("atom",),
            families=("periodic",),
        ),
    )
    library = build_atom_library(atoms, var_dims=(TARGET_DIM, TARGET_DIM, DIMLESS), y_dims=TARGET_DIM)
    blocks = seed_blocks_from_atom_relations(
        library,
        var_dims=(TARGET_DIM, TARGET_DIM, DIMLESS),
        required_dim=TARGET_DIM,
        limit=4,
    )
    product_expr = _expr(
        (
            "mul",
            ("sqrt", ("mul", ("var", 0), ("var", 1))),
            ("cos", ("var", 2)),
        )
    )

    assert product_expr in {str(node_str(block.node)) for block in blocks}
    assert any(block.source.startswith("aux:policy:") for block in blocks)


def test_aux_policy_plan_uses_protected_affine_relation_sweep():
    atoms = (
        EmergentAtom(
            node=simplify(("sqrt", ("mul", ("var", 0), ("var", 1)))),
            dim=TARGET_DIM,
            kind="carrier",
            score=2.0,
            evidence={"sources": ["unit"], "best_probe_gain_rel": 0.05},
            source_count=1,
            roles=("head:numerator",),
            families=("rational",),
        ),
        EmergentAtom(
            node=simplify(("cos", ("var", 2))),
            dim=DIMLESS,
            kind="dimensionless_feature",
            score=2.0,
            evidence={"sources": ["unit"], "best_parent_probe": 0.5},
            source_count=1,
            roles=("atom",),
            families=("periodic",),
        ),
    )
    library = build_atom_library(atoms, var_dims=(TARGET_DIM, TARGET_DIM, DIMLESS), y_dims=TARGET_DIM)
    plan = build_aux_policy_plan(
        families=("periodic", "power", "rational"),
        library=library,
        max_scaffolds=6,
        anchors_per_family=8,
    )

    assert plan == [
        {
            "family": "affine",
            "max_scaffolds": 6,
            "anchors_per_family": 8,
            "priority_score": plan[0]["priority_score"],
            "reason": "atom_policy_affine_relation_sweep",
        }
    ]
    assert plan[0]["priority_score"] > 0.0


def test_atomized_linear_span_solves_target_from_unit_compatible_atoms():
    atoms = (
        EmergentAtom(
            node=simplify(("add", ("var", 0), ("var", 1))),
            dim=TARGET_DIM,
            kind="target_term",
            score=5.0,
            evidence={"sources": ["unit"]},
            source_count=2,
            roles=("expr",),
            families=("affine",),
        ),
        EmergentAtom(
            node=simplify(("cos", ("var", 2))),
            dim=DIMLESS,
            kind="dimensionless_feature",
            score=6.0,
            evidence={"sources": ["unit"]},
            source_count=2,
            roles=("atom",),
            families=("periodic",),
        ),
        EmergentAtom(
            node=simplify(("mul", ("var", 0), ("var", 2))),
            dim=TARGET_DIM,
            kind="carrier",
            score=6.5,
            evidence={"sources": ["unit"]},
            source_count=2,
            roles=("head:numerator",),
            families=("rational",),
        ),
        EmergentAtom(
            node=simplify(("mul", ("var", 1), ("var", 2))),
            dim=TARGET_DIM,
            kind="carrier",
            score=6.4,
            evidence={"sources": ["unit"]},
            source_count=2,
            roles=("head:numerator",),
            families=("rational",),
        ),
        EmergentAtom(
            node=simplify(
                (
                    "mul",
                    ("add", ("var", 0), ("var", 1)),
                    ("cos", ("var", 2)),
                )
            ),
            dim=TARGET_DIM,
            kind="carrier",
            score=6.0,
            evidence={"sources": ["unit"]},
            source_count=2,
            roles=("head:numerator",),
            families=("rational",),
        ),
        EmergentAtom(
            node=simplify(("sqrt", ("var", 2))),
            dim=DIMLESS,
            kind="dimensionless_feature",
            score=5.8,
            evidence={"sources": ["unit"]},
            source_count=2,
            roles=("atom",),
            families=("power",),
        ),
        EmergentAtom(
            node=simplify(("sin", ("var", 2))),
            dim=DIMLESS,
            kind="dimensionless_feature",
            score=5.6,
            evidence={"sources": ["unit"]},
            source_count=2,
            roles=("atom",),
            families=("periodic",),
        ),
        EmergentAtom(
            node=simplify(("sqrt", ("mul", ("var", 0), ("var", 1)))),
            dim=TARGET_DIM,
            kind="carrier",
            score=1.0,
            evidence={"sources": ["unit"]},
            source_count=1,
            roles=("head:numerator",),
            families=("rational",),
        ),
    )
    x_fit, y_fit, x_probe, y_probe = _pb037_data()
    stats = {}
    rows = _propose_atomized_linear_span_rows(
        atoms=atoms,
        x_fit=x_fit,
        y_fit=y_fit,
        x_probe=x_probe,
        y_probe=y_probe,
        var_dims=(TARGET_DIM, TARGET_DIM, DIMLESS),
        y_dims=TARGET_DIM,
        max_depth=5,
        max_rows=16,
        stats=stats,
        debug_limit=4,
    )

    product_expr = _expr(
        (
            "mul",
            ("sqrt", ("mul", ("var", 0), ("var", 1))),
            ("cos", ("var", 2)),
        )
    )
    sum_expr = _expr(("add", ("var", 0), ("var", 1)))
    best = min(rows, key=lambda row: float(row["local_probe_mse"]))
    best_terms = set(best["direct_metadata"]["term_exprs"])

    assert float(best["local_probe_mse"]) < 1.0e-20
    assert sum_expr in best_terms
    assert product_expr in best_terms
    assert best["direct_metadata"]["include_intercept"] is False
    assert best["local_mapping_nparams"] == len(best["direct_metadata"]["term_exprs"])
    assert best["atomized_linear_span_atom_provenance"]
    assert stats["atomized_linear_span_rows"] > 0


def test_atomized_linear_span_structural_atoms_solve_pb037_without_observed_atoms():
    x_fit, y_fit, x_probe, y_probe = _pb037_data()
    stats = {}
    rows = _propose_atomized_linear_span_rows(
        atoms=(),
        x_fit=x_fit,
        y_fit=y_fit,
        x_probe=x_probe,
        y_probe=y_probe,
        var_dims=(TARGET_DIM, TARGET_DIM, DIMLESS),
        y_dims=TARGET_DIM,
        max_depth=5,
        max_rows=32,
        stats=stats,
        debug_limit=8,
    )

    product_expr = _expr(
        (
            "mul",
            ("sqrt", ("mul", ("var", 0), ("var", 1))),
            ("cos", ("var", 2)),
        )
    )
    sum_expr = _expr(("add", ("var", 0), ("var", 1)))
    best = min(rows, key=lambda row: float(row["local_probe_mse"]))
    best_terms = set(best["direct_metadata"]["term_exprs"])

    assert float(best["local_probe_mse"]) < 1.0e-20
    assert sum_expr in best_terms
    assert product_expr in best_terms
    assert stats["atomized_linear_span_structural_seed_target_atoms"] > 0
    assert stats["atomized_linear_span_structural_seed_dimless_atoms"] > 0


def test_atomized_linear_span_uses_raw_variables_and_dimless_closure_for_modulated_target():
    atoms = (
        EmergentAtom(
            node=simplify(("cos", ("mul", ("var", 1), ("var", 2)))),
            dim=DIMLESS,
            kind="dimensionless_feature",
            score=6.0,
            evidence={"sources": ["unit"]},
            source_count=2,
            roles=("atom",),
            families=("periodic",),
        ),
    )
    g = torch.Generator().manual_seed(50)
    x0 = 1.0 + 4.0 * torch.rand((240, 1), generator=g, dtype=torch.float64)
    x1 = 0.5 + 2.0 * torch.rand((240, 1), generator=g, dtype=torch.float64)
    x2 = 0.5 + 2.0 * torch.rand((240, 1), generator=g, dtype=torch.float64)
    x3 = 0.2 + 1.5 * torch.rand((240, 1), generator=g, dtype=torch.float64)
    x = torch.cat([x0, x1, x2, x3], dim=1)
    c = torch.cos(x1 * x2)
    y = x0 * (x3 * c.square() + c)
    x_fit, y_fit = x[:120], y[:120]
    x_probe, y_probe = x[120:], y[120:]
    y_dim = (1.0, 0.0, 0.0, 0.0, 0.0)
    stats = {}
    rows = _propose_atomized_linear_span_rows(
        atoms=atoms,
        x_fit=x_fit,
        y_fit=y_fit,
        x_probe=x_probe,
        y_probe=y_probe,
        var_dims=(y_dim, (0.0, -1.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0, 0.0), DIMLESS),
        y_dims=y_dim,
        max_depth=5,
        max_rows=32,
        stats=stats,
        debug_limit=8,
    )

    best = min(rows, key=lambda row: float(row["local_probe_mse"]))
    best_terms = set(best["direct_metadata"]["term_exprs"])

    assert float(best["local_probe_mse"]) < 1.0e-20
    assert _expr(("mul", ("cos", ("mul", ("var", 1), ("var", 2))), ("var", 0))) in best_terms
    assert any(
        expr in best_terms
        for expr in {
            _expr(
                (
                    "mul",
                    ("mul", ("var", 3), ("sqr", ("cos", ("mul", ("var", 1), ("var", 2))))),
                    ("var", 0),
                )
            ),
            _expr(
                (
                    "mul",
                    ("mul", ("var", 0), ("var", 3)),
                    ("sqr", ("cos", ("mul", ("var", 1), ("var", 2)))),
                )
            ),
        }
    )
    assert stats["atomized_linear_span_structural_seed_atoms"] >= 2
    assert stats["atomized_linear_span_derived_dimensionless_atoms"] > 0


def test_atom_policy_guides_affine_product_before_generic_pool():
    atoms = (
        EmergentAtom(
            node=simplify(("add", ("var", 0), ("var", 1))),
            dim=TARGET_DIM,
            kind="target_term",
            score=3.0,
            evidence={"sources": ["unit"], "best_probe_gain_rel": 0.2},
            source_count=1,
            roles=("expr",),
            families=("affine",),
        ),
        EmergentAtom(
            node=simplify(("sqrt", ("mul", ("var", 0), ("var", 1)))),
            dim=TARGET_DIM,
            kind="carrier",
            score=2.0,
            evidence={"sources": ["unit"], "best_probe_gain_rel": 0.05},
            source_count=1,
            roles=("head:numerator",),
            families=("rational",),
        ),
        EmergentAtom(
            node=simplify(("cos", ("var", 2))),
            dim=DIMLESS,
            kind="dimensionless_feature",
            score=2.0,
            evidence={"sources": ["unit"], "best_parent_probe": 0.5},
            source_count=1,
            roles=("atom",),
            families=("periodic",),
        ),
    )
    aux_blocks = seed_blocks_from_emergent_atoms(atoms, var_dims=(TARGET_DIM, TARGET_DIM, DIMLESS))
    library = build_atom_library(atoms, var_dims=(TARGET_DIM, TARGET_DIM, DIMLESS), y_dims=TARGET_DIM)
    apps = enumerate_operator_applications(
        families=("affine",),
        nvars=3,
        y_dims=TARGET_DIM,
        var_dims=(TARGET_DIM, TARGET_DIM, DIMLESS),
        pool_nodes=(),
        pool_dims=(),
        anchors_per_family=8,
        max_scaffolds=4,
        aux_seed_blocks=aux_blocks,
        atom_library=library,
        basis_seed_mode="basis_augmented",
    )

    sum_expr = _expr(("add", ("var", 0), ("var", 1)))
    product_expr = _expr(
        (
            "mul",
            ("sqrt", ("mul", ("var", 0), ("var", 1))),
            ("cos", ("var", 2)),
        )
    )
    binding_texts = [str(app.metadata.get("slot_bindings", "")) for app in apps]
    assert any(product_expr in text for text in binding_texts)
    assert any(sum_expr in text and product_expr in text for text in binding_texts)


def test_atom_policy_preserves_direct_relation_combo_when_products_are_crowded():
    atoms = (
        EmergentAtom(
            node=simplify(("add", ("var", 0), ("var", 1))),
            dim=TARGET_DIM,
            kind="target_term",
            score=4.0,
            evidence={"sources": ["unit"], "best_probe_gain_rel": 0.2},
            source_count=2,
            roles=("expr",),
            families=("affine",),
        ),
        EmergentAtom(
            node=simplify(("mul", ("var", 1), ("var", 2))),
            dim=TARGET_DIM,
            kind="carrier",
            score=5.0,
            evidence={"sources": ["unit"], "best_probe_gain_rel": 0.1, "best_parent_probe": 0.2},
            source_count=3,
            roles=("head:numerator",),
            families=("rational", "affine"),
        ),
        EmergentAtom(
            node=simplify(("mul", ("var", 0), ("var", 2))),
            dim=TARGET_DIM,
            kind="carrier",
            score=4.0,
            evidence={"sources": ["unit"], "best_probe_gain_rel": 0.08, "best_parent_probe": 0.3},
            source_count=2,
            roles=("head:numerator",),
            families=("rational", "affine"),
        ),
        EmergentAtom(
            node=simplify(("sqrt", ("mul", ("var", 0), ("var", 1)))),
            dim=TARGET_DIM,
            kind="carrier",
            score=1.0,
            evidence={"sources": ["unit"], "best_probe_gain_rel": 0.005, "best_parent_probe": 1.0},
            source_count=1,
            roles=("head:numerator",),
            families=("rational",),
        ),
        EmergentAtom(
            node=simplify(("cos", ("var", 2))),
            dim=DIMLESS,
            kind="dimensionless_feature",
            score=3.0,
            evidence={"sources": ["unit"], "best_parent_probe": 0.4},
            source_count=2,
            roles=("atom",),
            families=("periodic",),
        ),
        EmergentAtom(
            node=simplify(("sin", ("var", 2))),
            dim=DIMLESS,
            kind="dimensionless_feature",
            score=3.0,
            evidence={"sources": ["unit"], "best_parent_probe": 0.4},
            source_count=2,
            roles=("atom",),
            families=("periodic",),
        ),
    )
    aux_blocks = seed_blocks_from_emergent_atoms(atoms, var_dims=(TARGET_DIM, TARGET_DIM, DIMLESS))
    library = build_atom_library(atoms, var_dims=(TARGET_DIM, TARGET_DIM, DIMLESS), y_dims=TARGET_DIM)
    apps = enumerate_operator_applications(
        families=("affine",),
        nvars=3,
        y_dims=TARGET_DIM,
        var_dims=(TARGET_DIM, TARGET_DIM, DIMLESS),
        pool_nodes=(),
        pool_dims=(),
        anchors_per_family=8,
        max_scaffolds=12,
        aux_seed_blocks=aux_blocks,
        atom_library=library,
        basis_seed_mode="basis_augmented",
    )

    sum_expr = _expr(("add", ("var", 0), ("var", 1)))
    product_expr = _expr(
        (
            "mul",
            ("sqrt", ("mul", ("var", 0), ("var", 1))),
            ("cos", ("var", 2)),
        )
    )
    binding_texts = [str(app.metadata.get("slot_bindings", "")) for app in apps]

    assert any(sum_expr in text and product_expr in text for text in binding_texts)


def test_atom_policy_guides_periodic_carrier_and_envelope_slots():
    atoms = (
        EmergentAtom(
            node=simplify(("sqrt", ("mul", ("var", 0), ("var", 1)))),
            dim=TARGET_DIM,
            kind="carrier",
            score=2.0,
            evidence={"sources": ["unit"], "best_probe_gain_rel": 0.05},
            source_count=1,
            roles=("head:numerator",),
            families=("rational",),
        ),
        EmergentAtom(
            node=simplify(("cos", ("var", 2))),
            dim=DIMLESS,
            kind="dimensionless_feature",
            score=2.0,
            evidence={"sources": ["unit"], "best_parent_probe": 0.5},
            source_count=1,
            roles=("atom",),
            families=("periodic",),
        ),
    )
    aux_blocks = seed_blocks_from_emergent_atoms(atoms, var_dims=(TARGET_DIM, TARGET_DIM, DIMLESS))
    library = build_atom_library(atoms, var_dims=(TARGET_DIM, TARGET_DIM, DIMLESS), y_dims=TARGET_DIM)
    apps = enumerate_operator_applications(
        families=("periodic",),
        nvars=3,
        y_dims=TARGET_DIM,
        var_dims=(TARGET_DIM, TARGET_DIM, DIMLESS),
        pool_nodes=(),
        pool_dims=(),
        anchors_per_family=8,
        max_scaffolds=40,
        aux_seed_blocks=aux_blocks,
        atom_library=library,
        basis_seed_mode="basis_augmented",
    )

    sqrt_expr = _expr(("sqrt", ("mul", ("var", 0), ("var", 1))))
    binding_texts = [
        str(app.metadata.get("slot_bindings", ""))
        for app in apps
        if app.operator_id == "periodic:cos_mul"
    ]
    assert any(sqrt_expr in text and "carrier" in text and "x2" in text for text in binding_texts)


def test_materialized_linear_combo_keeps_embedded_bias():
    expr = materialize_direct_linear_combo(
        [(2.0, ("var", 0))],
        bias=3.0,
        embed_coefficients=True,
    )

    assert "3" in str(node_str(expr))
    assert "2" in str(node_str(expr))
    assert "x0" in str(node_str(expr))


def test_simplify_does_not_apply_unsafe_domain_rewrites():
    cases = [
        ("sqrt", ("sqr", ("var", 0))),
        ("sqrt", ("exp", ("var", 0))),
        ("log", ("sqr", ("var", 0))),
        ("sin", ("cos", ("var", 0))),
    ]

    for node in cases:
        assert simplify(node) == node
