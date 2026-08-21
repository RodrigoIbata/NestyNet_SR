# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

from __future__ import annotations

import torch

from nestynet_sr.sr_search.factorized_search.basis_scoring import make_additive_basis_transition
from nestynet_sr.sr_search.factorized_search.basis_state import (
    ProposalContext,
    admit_basis_state_to_beam,
    basis_state_from_closure_candidate,
)
from nestynet_sr.sr_search.factorized_search.engine.proposal_execution import (
    ProposalScoringState,
    merge_route_status_counts,
    run_closure_search_pass,
    score_native_candidate_basis_state,
    score_external_candidate_expr,
)


def _base_closure_search_kwargs():
    return {
        "closure_search_enable": True,
        "closure_search_stats": {},
        "closure_search_families": ["periodic"],
        "closure_search_max_proposals": 8,
        "closure_search_anchors_per_family": 4,
        "closure_search_preview_topk": 4,
        "closure_search_exact_topk": 2,
        "closure_search_beam_width": 4,
        "closure_search_seed_exact_topk": 6,
        "closure_search_seed_beam_width": 4,
        "closure_search_seed_scaffold_reserve": 0,
        "closure_search_seed_family_cap": 2,
        "closure_search_seed_exact_bound_bonus": 0.25,
        "closure_search_pair_normal_enable": False,
        "closure_search_pair_normal_topk": 3,
        "closure_search_pair_normal_max_pairs": 1,
        "closure_search_min_valid_frac": 0.1,
        "closure_search_min_confidence": 0.1,
        "closure_search_periodic_min_valid_scale": 1.0,
        "closure_search_periodic_min_confidence_scale": 1.0,
        "closure_search_transport_min_lin_rel": 0.0,
        "inverse_periodic_path_penalty": 0.0,
        "inverse_nonperiodic_muldiv_bonus": 0.0,
        "inverse_nonperiodic_explogsqrt_bonus": 0.0,
        "inverse_branch_beam_width": 4,
        "inverse_topk_terms": 4,
        "inverse_shortlist_mult": 1,
        "inverse_local_score_mode": "mse",
        "inverse_micro_search_enable": False,
        "inverse_micro_search_max_depth": 2,
        "inverse_micro_search_beam_width": 2,
        "inverse_micro_search_topk": 2,
        "inverse_micro_search_seed_terms": 2,
        "inverse_target_mode": "full",
        "inverse_safe_eps": 1.0e-8,
        "inverse_confidence_mode": "gain",
        "inverse_confidence_target_gain": 0.0,
        "inverse_confidence_floor": 0.0,
        "inverse_full_mapping_penalty": 0.0,
        "inverse_exact_simple_target_bonus": 0.0,
        "inverse_additive_descend_penalty": 0.0,
        "inverse_nonadditive_leaf_penalty": 0.0,
        "inverse_exact_path_eta": 0.0,
        "inverse_branch_ambiguity_penalty": 0.0,
        "inverse_transport_min_effective_n": 0.0,
        "inverse_spec_regime_metadata": None,
        "inverse_spec_local_score_mode": "mse",
        "inverse_spec_enum_max_depth": 2,
        "inverse_spec_enum_max_trees": 16,
        "inverse_spec_max_subtree_depth": None,
        "inverse_spec_complexity_penalty": 0.0,
        "inverse_spec_family_battery_enable": False,
        "inverse_spec_family_battery_mode": "outer",
        "inverse_spec_recursive_enable": False,
        "inverse_spec_recursive_max_depth": 2,
        "inverse_spec_recursive_trigger_rel_mse": 0.0,
        "inverse_spec_recursive_seed_cap": 2,
        "inverse_spec_recursive_branch_topk": 2,
        "inverse_spec_recursive_child_topk": 2,
        "inverse_spec_witness_jets_enable": False,
        "inverse_spec_witness_d2_enable": False,
        "inverse_spec_witness_max_rows": 16,
        "inverse_spec_witness_loss_enable": False,
        "inverse_spec_witness_grad_weight": 0.0,
        "inverse_spec_witness_d2_weight": 0.0,
        "inverse_spec_witness_diag_weight": 0.0,
        "inverse_spec_witness_physics_weight": 0.0,
        "inverse_spec_active_var_screen_enable": False,
        "inverse_spec_active_var_grad_tol": 0.0,
        "inverse_spec_active_var_max_count": 0,
        "wall_time_deadline": None,
        "wall_time_limit_s": None,
        "max_depth": 5,
        "poly_degree": 2,
        "nvars": 4,
        "x_fit": None,
        "y_fit": None,
        "x_probe": None,
        "y_probe": None,
        "var_dims": None,
        "y_dims": None,
        "boost_pool_nodes": [],
        "boost_pool_phi_fit": None,
        "boost_pool_phi": None,
        "boost_pool_dims": [],
        "dm": False,
        "wall_time_exceeded_fn": lambda: False,
        "node_str_fn": lambda expr: str(expr),
    }


def test_merge_route_status_counts_accumulates_positive_entries():
    stats = {"status_counts": {"ok": 1}}
    merge_route_status_counts(stats, {"ok": 2, "skip": 3, "zero": 0, "bad": "x"})
    assert stats["status_counts"] == {"ok": 3, "skip": 3}


def test_run_closure_search_pass_scores_best_per_scaffold_and_dedupes_children():
    scored = []

    def _run_closure_search_pass_impl(**_kwargs):
        return {
            "stats": {
                "families_considered": 1,
                "preview_calls": 3,
                "status_counts": {"ok": 3},
            },
            "candidate_rows": [
                {
                    "scaffold_id": "s1",
                    "expr": ("var", 0),
                    "child_key": "dup",
                    "local_probe_mse": 0.20,
                    "local_fit_mse": 0.20,
                    "candidate_child_size": 1,
                },
                {
                    "scaffold_id": "s1",
                    "expr": ("var", 1),
                    "child_key": "dup",
                    "local_probe_mse": 0.10,
                    "local_fit_mse": 0.10,
                    "candidate_child_size": 1,
                },
                {
                    "scaffold_id": "s2",
                    "expr": ("var", 2),
                    "child_key": "dup",
                    "local_probe_mse": 0.05,
                    "local_fit_mse": 0.05,
                    "candidate_child_size": 1,
                },
                {
                    "scaffold_id": "s3",
                    "expr": ("var", 3),
                    "child_key": "uniq",
                    "local_probe_mse": 0.30,
                    "local_fit_mse": 0.30,
                    "candidate_child_size": 1,
                },
            ],
        }

    def _score_external_candidate_expr_fn(expr, **kwargs):
        scored.append((expr, kwargs["candidate_meta"]["scaffold_id"]))

    kwargs = _base_closure_search_kwargs()
    kwargs.update(
        {
            "run_closure_search_pass_impl": _run_closure_search_pass_impl,
            "score_external_candidate_expr_fn": _score_external_candidate_expr_fn,
        }
    )

    run_closure_search_pass(**kwargs)

    assert scored == [(("var", 2), "s2"), (("var", 3), "s3")]
    assert kwargs["closure_search_stats"]["families_considered"] == 1
    assert kwargs["closure_search_stats"]["preview_calls"] == 3
    assert kwargs["closure_search_stats"]["status_counts"]["ok"] == 3
    assert kwargs["closure_search_stats"]["family_steering_applied"] is False
    assert kwargs["closure_search_stats"]["proposal_context"]["total_budget"] == 8


def test_run_closure_search_pass_marks_deadline_before_preview():
    kwargs = _base_closure_search_kwargs()
    kwargs.update(
        {
            "wall_time_exceeded_fn": lambda: True,
            "run_closure_search_pass_impl": lambda **_kwargs: (_ for _ in ()).throw(
                AssertionError("preview runner should not be called")
            ),
            "score_external_candidate_expr_fn": lambda *args, **kwargs: None,
        }
    )

    run_closure_search_pass(**kwargs)

    stats = kwargs["closure_search_stats"]
    assert stats["deadline_exceeded"] is True
    assert stats["wall_time_budget_s"] == 0.0
    assert stats["status_counts"]["deadline_exceeded"] == 1


def test_run_closure_search_pass_accepts_explicit_proposal_context():
    seen = {}

    def _run_closure_search_pass_impl(**kwargs):
        seen["proposal_context"] = kwargs.get("proposal_context")
        return {"stats": {}, "candidate_rows": []}

    kwargs = _base_closure_search_kwargs()
    kwargs.update(
        {
            "proposal_context": ProposalContext(
                residual_witness="oscillatory residual",
                family_hints={"periodic": 4.0},
                total_budget=3,
            ),
            "run_closure_search_pass_impl": _run_closure_search_pass_impl,
            "score_external_candidate_expr_fn": lambda *args, **kwargs: None,
        }
    )

    run_closure_search_pass(**kwargs)

    proposal_context = seen["proposal_context"]
    assert isinstance(proposal_context, ProposalContext)
    assert proposal_context.family_hints["periodic"] == 4.0


def test_run_closure_search_pass_prefers_native_basis_state_scoring_for_native_candidates():
    expr = ("add", ("var", 0), ("var", 1))
    state = basis_state_from_closure_candidate(
        family="affine",
        scaffold_id="affine:latent:test",
        expr=expr,
        anchor_node=None,
        scaffold_metadata={},
        local_fit_mse=0.0,
        local_probe_mse=0.0,
        local_mapping_kind="direct_affine_head",
        local_mapping_coeffs=[1.0, 1.0, 0.0],
    )

    def _run_closure_search_pass_impl(**_kwargs):
        return {
            "stats": {},
            "candidate_rows": [
                {
                    "scaffold_id": "affine:latent:test",
                    "expr": expr,
                    "child_key": "((x0)+(x1))",
                    "proposal_key": "native-affine-demo",
                    "local_probe_mse": 0.05,
                    "local_fit_mse": 0.05,
                    "candidate_child_size": 3,
                    "basis_state_obj": state,
                    "basis_state_dict": state.to_dict(),
                }
            ],
        }

    kwargs = _base_closure_search_kwargs()
    x = torch.tensor(
        [
            [0.2, 0.3],
            [0.4, 0.5],
            [0.6, 0.7],
            [0.8, 0.9],
        ],
        dtype=torch.float64,
    )
    y = (x[:, :1] + x[:, 1:2])
    scoring_state = ProposalScoringState(
        n_evaluated=0,
        best_raw_mse=float("inf"),
        best_raw_mse_struct=float("inf"),
        best_mse=float("inf"),
    )
    seen = {}

    class _Arch:
        def update(self, key, mse_eff, expr, z_col, mapping, raw_mse=None):
            seen["mapping"] = mapping
            return True

    kwargs.update(
        {
            "closure_search_families": ["affine"],
            "nvars": 2,
            "x_fit": x,
            "y_fit": y,
            "x_probe": x,
            "y_probe": y,
            "run_closure_search_pass_impl": _run_closure_search_pass_impl,
            "score_external_candidate_expr_fn": lambda *args, **kwargs: (_ for _ in ()).throw(
                AssertionError("legacy expression scorer should not be used for native basis candidates")
            ),
            "score_native_candidate_basis_state_fn": lambda **inner_kwargs: score_native_candidate_basis_state(
                **inner_kwargs,
                state=scoring_state,
                x_fit=x,
                y_fit=y,
                x_probe=x,
                y_probe=y,
                complexity_penalty=0.0,
                node_str_fn=lambda node: str(node),
                arch=_Arch(),
            ),
            "node_str_fn": lambda node: str(node),
        }
    )

    run_closure_search_pass(**kwargs)

    stats = kwargs["closure_search_stats"]
    assert int(stats["basis_state_native_scores"]) >= 1
    assert stats["basis_state_beam_count"] >= 1
    assert stats["basis_state_controller_stop_reason"] in {"no_basis_update", "round_limit", "scaffold_budget_exhausted"}
    mapping = seen["mapping"]
    assert mapping["_acceptance_basis"] == "basis_state_native"
    assert mapping["_score_ladder"]["compiled_structural"]["accepted"] is True
    assert mapping["_score_ladder"]["head_augmented"]["accepted"] is True
    assert mapping["_basis_state_summary"]["block_count"] >= 1


def test_run_closure_search_pass_fasttrack_native_preserves_preview_basis_state():
    wrong_expr = ("sqrt", ("add", ("sqr", ("mul", ("var", 0), ("var", 1))), ("sqr", ("mul", ("var", 0), ("var", 2)))))
    exact_expr = (
        "mul",
        ("var", 0),
        ("sqrt", ("add", ("add", ("sqr", ("var", 1)), ("sqr", ("var", 2))), ("sqr", ("var", 3)))),
    )
    base_state = basis_state_from_closure_candidate(
        family="quadratic",
        scaffold_id="quadratic:wrong",
        expr=wrong_expr,
        anchor_node=None,
        scaffold_metadata={"form": "quadratic_sqrt"},
        local_fit_mse=6.0,
        local_probe_mse=6.0,
        local_mapping_kind="direct_quadratic_head",
        direct_metadata={"quadratic_base_nodes": [("mul", ("var", 0), ("var", 1)), ("mul", ("var", 0), ("var", 2))]},
    )
    preview_state = basis_state_from_closure_candidate(
        family="quadratic",
        scaffold_id="quadratic:exact",
        expr=exact_expr,
        anchor_node=("var", 0),
        scaffold_metadata={"form": "quadratic_sqrt_mul"},
        local_fit_mse=1.0e-12,
        local_probe_mse=1.0e-12,
        local_mapping_kind="direct_quadratic_head",
        direct_metadata={"quadratic_base_nodes": [("var", 1), ("var", 2), ("var", 3)]},
    )

    def _run_closure_search_pass_impl(**_kwargs):
        return {
            "stats": {},
            "candidate_rows": [
                {
                    "scaffold_id": "quadratic:exact",
                    "expr": exact_expr,
                    "proposal_key": "quadratic:exact",
                    "child_key": "quadratic:exact",
                    "local_probe_mse": 1.0e-12,
                    "local_fit_mse": 1.0e-12,
                    "candidate_child_size": 4,
                    "preview_fasttrack": True,
                    "basis_state_obj": preview_state,
                    "basis_state_dict": preview_state.to_dict(),
                    "feature_block_obj": preview_state.blocks[0],
                    "proposal_lane": "core",
                    "scaffold_family": "quadratic",
                    "operator_id": "quadratic:sqrt_mul",
                }
            ],
        }

    x = torch.tensor(
        [
            [0.6, 0.7, 0.8, 0.9],
            [1.0, 1.1, 1.2, 1.3],
            [1.4, 1.5, 1.6, 1.7],
            [1.8, 1.9, 2.0, 2.1],
        ],
        dtype=torch.float64,
    )
    y = x[:, :1] * torch.sqrt(x[:, 1:2].square() + x[:, 2:3].square() + x[:, 3:4].square())
    scoring_state = ProposalScoringState(
        n_evaluated=0,
        best_raw_mse=float("inf"),
        best_raw_mse_struct=float("inf"),
        best_mse=float("inf"),
    )
    seen = {}

    class _Arch:
        def update(self, key, mse_eff, expr, z_col, mapping, raw_mse=None):
            seen["expr"] = expr
            seen["raw_mse"] = raw_mse
            return True

    def _score_native(**inner_kwargs):
        prepared_state = inner_kwargs["candidate_meta"]["basis_state_obj"]
        seen["prepared_block_count"] = len(tuple(getattr(prepared_state, "blocks", ()) or ()))
        seen["prepared_expr"] = getattr(prepared_state, "compiled_expr", None)
        seen["prepare_mode"] = inner_kwargs["candidate_meta"].get("basis_state_prepare_mode")
        seen["preserve"] = bool(inner_kwargs["candidate_meta"].get("basis_state_direct_preserve", False))
        return score_native_candidate_basis_state(
            **inner_kwargs,
            state=scoring_state,
            x_fit=x,
            y_fit=y,
            x_probe=x,
            y_probe=y,
            complexity_penalty=0.0,
            node_str_fn=lambda node: str(node),
            arch=_Arch(),
        )

    kwargs = _base_closure_search_kwargs()
    kwargs.update(
        {
            "closure_search_families": ["quadratic"],
            "x_fit": x,
            "y_fit": y,
            "x_probe": x,
            "y_probe": y,
            "proposal_context": ProposalContext(
                basis_state=base_state,
                basis_state_beam=admit_basis_state_to_beam([], base_state, beam_width=3),
                total_budget=2,
            ),
            "run_closure_search_pass_impl": _run_closure_search_pass_impl,
            "score_external_candidate_expr_fn": lambda *args, **kwargs: (_ for _ in ()).throw(
                AssertionError("fast-track native candidate should not use external scorer")
            ),
            "score_native_candidate_basis_state_fn": _score_native,
            "node_str_fn": lambda node: str(node),
        }
    )

    run_closure_search_pass(**kwargs)

    stats = kwargs["closure_search_stats"]
    assert seen["preserve"] is True
    assert seen["prepare_mode"] == "preview_only"
    assert int(seen["prepared_block_count"]) == 1
    assert seen["prepared_expr"] == preview_state.compiled_expr
    assert seen["expr"] == preview_state.compiled_expr
    assert float(seen["raw_mse"]) <= 1.0e-12
    assert int(stats.get("basis_state_native_preserved", 0) or 0) >= 1


def test_run_closure_search_pass_builds_residual_guided_context_and_hints():
    seen = {}

    def _run_closure_search_pass_impl(**kwargs):
        seen["proposal_context"] = kwargs.get("proposal_context")
        return {"stats": {}, "candidate_rows": []}

    x = torch.linspace(0.0, 3.14159, steps=64, dtype=torch.float64).unsqueeze(1)
    y = torch.sin(x)

    kwargs = _base_closure_search_kwargs()
    kwargs.update(
        {
            "closure_search_families": ["periodic", "exp", "power"],
            "nvars": 1,
            "x_fit": x,
            "y_fit": y,
            "x_probe": x,
            "y_probe": y,
            "boost_pool_nodes": [("var", 0)],
            "boost_pool_phi_fit": torch.zeros((64, 1), dtype=torch.float64),
            "boost_pool_phi": torch.zeros((64, 1), dtype=torch.float64),
            "run_closure_search_pass_impl": _run_closure_search_pass_impl,
            "score_external_candidate_expr_fn": lambda *args, **kwargs: None,
        }
    )

    run_closure_search_pass(**kwargs)

    proposal_context = seen["proposal_context"]
    assert isinstance(proposal_context, ProposalContext)
    assert isinstance(proposal_context.residual_witness, dict)
    assert float(proposal_context.family_hints.get("periodic", 0.0)) > 0.5
    assert float(proposal_context.family_hints.get("periodic", 0.0)) > float(
        proposal_context.family_hints.get("exp", 0.0)
    )
    assert proposal_context.diagnostics["route"] == "closure_search"


def test_run_closure_search_pass_maintains_basis_state_beam_and_skips_covered_candidates():
    base_state = basis_state_from_closure_candidate(
        family="periodic",
        scaffold_id="periodic:cos",
        expr=("cos", ("var", 0)),
        anchor_node=None,
        scaffold_metadata={"form": "cos_base"},
        local_fit_mse=0.2,
        local_probe_mse=0.2,
        local_mapping_kind="direct_linear_head",
        direct_metadata={"hole_node": ("var", 0), "feature_node": ("cos", ("var", 0))},
    )
    new_state = basis_state_from_closure_candidate(
        family="power",
        scaffold_id="power:sqrt",
        expr=("sqrt", ("var", 1)),
        anchor_node=None,
        scaffold_metadata={"form": "power_sqrt"},
        local_fit_mse=0.1,
        local_probe_mse=0.1,
        local_mapping_kind="direct_power_head",
        direct_metadata={"hole_node": ("var", 1), "power_inner_node": ("var", 1)},
    )
    scored = []

    def _run_closure_search_pass_impl(**_kwargs):
        return {
            "stats": {},
            "candidate_rows": [
                {
                    "scaffold_id": "covered",
                    "expr": ("cos", ("var", 0)),
                    "child_key": "covered",
                    "local_probe_mse": 0.01,
                    "local_fit_mse": 0.01,
                    "candidate_child_size": 2,
                    "basis_state_obj": base_state,
                    "feature_block_obj": base_state.blocks[0],
                },
                {
                    "scaffold_id": "new",
                    "expr": ("sqrt", ("var", 1)),
                    "child_key": "new",
                    "local_probe_mse": 0.02,
                    "local_fit_mse": 0.02,
                    "candidate_child_size": 2,
                    "basis_state_obj": new_state,
                    "feature_block_obj": new_state.blocks[0],
                },
            ],
        }

    def _score_external_candidate_expr_fn(expr, **kwargs):
        scored.append((expr, kwargs["candidate_meta"]["scaffold_id"]))
        return {
            "expr": expr,
            "raw_mse": 0.01,
            "basis_state_obj": kwargs["candidate_meta"]["basis_state_obj"],
        }

    kwargs = _base_closure_search_kwargs()
    kwargs.update(
        {
            "proposal_context": ProposalContext(
                basis_state=base_state,
                basis_state_beam=admit_basis_state_to_beam([], base_state, beam_width=2),
                total_budget=2,
            ),
            "run_closure_search_pass_impl": _run_closure_search_pass_impl,
            "score_external_candidate_expr_fn": _score_external_candidate_expr_fn,
        }
    )

    run_closure_search_pass(**kwargs)

    assert scored == [(("sqrt", ("var", 1)), "new")]
    stats = kwargs["closure_search_stats"]
    assert int(stats["basis_state_skip_covered"]) >= 1
    assert int(stats["basis_state_beam_count"]) >= 1
    assert len(list(stats["basis_state_beam"] or [])) >= 1


def test_run_closure_search_pass_scores_prepared_basis_state_candidate():
    base_state = basis_state_from_closure_candidate(
        family="periodic",
        scaffold_id="periodic:cos",
        expr=("cos", ("var", 0)),
        anchor_node=None,
        scaffold_metadata={"form": "cos_base"},
        local_fit_mse=0.2,
        local_probe_mse=0.2,
        local_mapping_kind="direct_linear_head",
        direct_metadata={"hole_node": ("var", 0), "feature_node": ("cos", ("var", 0))},
    )
    preview_state = basis_state_from_closure_candidate(
        family="power",
        scaffold_id="power:sqrt",
        expr=("sqrt", ("var", 1)),
        anchor_node=None,
        scaffold_metadata={"form": "power_sqrt"},
        local_fit_mse=0.1,
        local_probe_mse=0.1,
        local_mapping_kind="direct_power_head",
        direct_metadata={"hole_node": ("var", 1), "power_inner_node": ("var", 1)},
    )
    seen = {}

    def _run_closure_search_pass_impl(**_kwargs):
        return {
            "stats": {},
            "candidate_rows": [
                {
                    "scaffold_id": "new",
                    "expr": ("sqrt", ("var", 1)),
                    "child_key": "new",
                    "local_probe_mse": 0.02,
                    "local_fit_mse": 0.02,
                    "candidate_child_size": 2,
                    "basis_state_obj": preview_state,
                    "feature_block_obj": preview_state.blocks[0],
                },
            ],
        }

    def _score_external_candidate_expr_fn(expr, **kwargs):
        seen["candidate_basis_state"] = kwargs["candidate_meta"]["basis_state_obj"]
        return {
            "expr": expr,
            "raw_mse": 0.01,
            "basis_state_obj": kwargs["candidate_meta"]["basis_state_obj"],
        }

    kwargs = _base_closure_search_kwargs()
    kwargs.update(
        {
            "proposal_context": ProposalContext(
                basis_state=base_state,
                basis_state_beam=admit_basis_state_to_beam([], base_state, beam_width=2),
                total_budget=2,
            ),
            "run_closure_search_pass_impl": _run_closure_search_pass_impl,
            "score_external_candidate_expr_fn": _score_external_candidate_expr_fn,
        }
    )

    run_closure_search_pass(**kwargs)

    candidate_state = seen["candidate_basis_state"]
    assert candidate_state is not None
    assert candidate_state.to_dict()["block_count"] == 2
    assert candidate_state.to_dict()["compiled_expr"] == "sqrt(x1)"


def test_run_closure_search_pass_iterates_after_basis_state_admission():
    base_state = basis_state_from_closure_candidate(
        family="periodic",
        scaffold_id="periodic:cos",
        expr=("cos", ("var", 0)),
        anchor_node=None,
        scaffold_metadata={"form": "cos_base"},
        local_fit_mse=0.3,
        local_probe_mse=0.3,
        local_mapping_kind="direct_linear_head",
        direct_metadata={"hole_node": ("var", 0), "feature_node": ("cos", ("var", 0))},
    )
    preview_state_round1 = basis_state_from_closure_candidate(
        family="power",
        scaffold_id="power:sqrt",
        expr=("sqrt", ("var", 1)),
        anchor_node=None,
        scaffold_metadata={"form": "power_sqrt"},
        local_fit_mse=0.2,
        local_probe_mse=0.2,
        local_mapping_kind="direct_power_head",
        direct_metadata={"hole_node": ("var", 1), "power_inner_node": ("var", 1)},
    )
    preview_state_round2 = basis_state_from_closure_candidate(
        family="log",
        scaffold_id="log:base",
        expr=("log", ("var", 2)),
        anchor_node=None,
        scaffold_metadata={"form": "log_base"},
        local_fit_mse=0.1,
        local_probe_mse=0.1,
        local_mapping_kind="direct_linear_head",
        direct_metadata={"hole_node": ("var", 2), "feature_node": ("log", ("var", 2))},
    )
    call_log = []

    def _run_closure_search_pass_impl(**kwargs):
        round_idx = len(call_log)
        proposal_context = kwargs.get("proposal_context")
        basis_state = getattr(proposal_context, "basis_state", None)
        call_log.append(
            {
                "block_count": int(getattr(basis_state, "block_count", lambda: 0)())
                if basis_state is not None
                else 0,
                "beam_count": len(tuple(getattr(proposal_context, "basis_state_beam", ()) or ())),
            }
        )
        if round_idx == 0:
            return {
                "stats": {},
                "candidate_rows": [
                    {
                        "scaffold_id": "round1",
                        "expr": ("sqrt", ("var", 1)),
                        "child_key": "round1",
                        "local_probe_mse": 0.01,
                        "local_fit_mse": 0.01,
                        "candidate_child_size": 2,
                        "basis_state_obj": preview_state_round1,
                        "feature_block_obj": preview_state_round1.blocks[0],
                    },
                ],
            }
        return {
            "stats": {},
            "candidate_rows": [
                {
                    "scaffold_id": "round2",
                    "expr": ("log", ("var", 2)),
                    "child_key": "round2",
                    "local_probe_mse": 0.01,
                    "local_fit_mse": 0.01,
                    "candidate_child_size": 2,
                    "basis_state_obj": preview_state_round2,
                    "feature_block_obj": preview_state_round2.blocks[0],
                },
            ],
        }

    def _score_external_candidate_expr_fn(expr, **kwargs):
        candidate_state = kwargs["candidate_meta"]["basis_state_obj"]
        raw_mse = 0.02 if expr == ("sqrt", ("var", 1)) else 0.01
        return {
            "expr": expr,
            "raw_mse": raw_mse,
            "basis_state_obj": candidate_state,
        }

    kwargs = _base_closure_search_kwargs()
    kwargs.update(
        {
            "proposal_context": ProposalContext(
                basis_state=base_state,
                basis_state_beam=admit_basis_state_to_beam([], base_state, beam_width=3),
                total_budget=3,
            ),
            "run_closure_search_pass_impl": _run_closure_search_pass_impl,
            "score_external_candidate_expr_fn": _score_external_candidate_expr_fn,
        }
    )

    run_closure_search_pass(**kwargs)

    stats = kwargs["closure_search_stats"]
    assert len(call_log) == 2
    assert int(call_log[0]["beam_count"]) >= 1
    assert int(call_log[1]["beam_count"]) >= 2
    assert int(stats["closure_search_rounds"]) == 2
    assert int(stats["basis_state_round_updates"]) == 2
    assert int(stats["basis_state_beam_count"]) >= 1
    assert any(int(row.get("block_count", 0) or 0) >= 3 for row in list(stats["basis_state_beam"] or []))


def test_run_closure_search_pass_rejects_basis_state_without_probe_gain():
    base_state = basis_state_from_closure_candidate(
        family="periodic",
        scaffold_id="periodic:cos",
        expr=("cos", ("var", 0)),
        anchor_node=None,
        scaffold_metadata={"form": "cos_base"},
        local_fit_mse=0.1,
        local_probe_mse=0.1,
        local_mapping_kind="direct_linear_head",
        direct_metadata={"hole_node": ("var", 0), "feature_node": ("cos", ("var", 0))},
    )
    preview_state = basis_state_from_closure_candidate(
        family="power",
        scaffold_id="power:sqrt",
        expr=("sqrt", ("var", 1)),
        anchor_node=None,
        scaffold_metadata={"form": "power_sqrt"},
        local_fit_mse=0.2,
        local_probe_mse=0.2,
        local_mapping_kind="direct_power_head",
        direct_metadata={"hole_node": ("var", 1), "power_inner_node": ("var", 1)},
    )

    def _run_closure_search_pass_impl(**_kwargs):
        return {
            "stats": {},
            "candidate_rows": [
                {
                    "scaffold_id": "candidate",
                    "expr": ("sqrt", ("var", 1)),
                    "child_key": "candidate",
                    "local_probe_mse": 0.02,
                    "local_fit_mse": 0.02,
                    "candidate_child_size": 2,
                    "basis_state_obj": preview_state,
                    "feature_block_obj": preview_state.blocks[0],
                },
            ],
        }

    def _score_external_candidate_expr_fn(expr, **kwargs):
        candidate_state = kwargs["candidate_meta"]["basis_state_obj"]
        return {
            "expr": expr,
            "raw_mse": 0.1,
            "fit_loss": 0.1,
            "probe_loss": 0.1,
            "basis_state_obj": candidate_state,
        }

    kwargs = _base_closure_search_kwargs()
    kwargs.update(
        {
            "proposal_context": ProposalContext(
                basis_state=base_state,
                basis_state_beam=admit_basis_state_to_beam([], base_state, beam_width=3),
                total_budget=2,
            ),
            "run_closure_search_pass_impl": _run_closure_search_pass_impl,
            "score_external_candidate_expr_fn": _score_external_candidate_expr_fn,
        }
    )

    run_closure_search_pass(**kwargs)

    stats = kwargs["closure_search_stats"]
    assert int(stats["basis_state_accept_attempts"]) >= 1
    assert int(stats["basis_state_accept_rejected"]) >= 1
    assert int(stats["basis_state_round_updates"]) == 0
    assert str(stats["basis_state_controller_stop_reason"]) == "no_basis_update"


def test_run_closure_search_pass_scores_full_round_before_basis_restart():
    base_state = basis_state_from_closure_candidate(
        family="periodic",
        scaffold_id="periodic:cos",
        expr=("cos", ("var", 0)),
        anchor_node=None,
        scaffold_metadata={"form": "cos_base"},
        local_fit_mse=0.3,
        local_probe_mse=0.3,
        local_mapping_kind="direct_linear_head",
        direct_metadata={"hole_node": ("var", 0), "feature_node": ("cos", ("var", 0))},
    )
    preview_state_round1 = basis_state_from_closure_candidate(
        family="power",
        scaffold_id="power:sqrt",
        expr=("sqrt", ("var", 1)),
        anchor_node=None,
        scaffold_metadata={"form": "power_sqrt"},
        local_fit_mse=0.2,
        local_probe_mse=0.2,
        local_mapping_kind="direct_power_head",
        direct_metadata={"hole_node": ("var", 1), "power_inner_node": ("var", 1)},
    )
    stale_same_round_state = basis_state_from_closure_candidate(
        family="exp",
        scaffold_id="exp:base",
        expr=("exp", ("var", 2)),
        anchor_node=None,
        scaffold_metadata={"form": "exp_base"},
        local_fit_mse=0.15,
        local_probe_mse=0.15,
        local_mapping_kind="direct_linear_head",
        direct_metadata={"hole_node": ("var", 2), "feature_node": ("exp", ("var", 2))},
    )
    preview_state_round2 = basis_state_from_closure_candidate(
        family="log",
        scaffold_id="log:base",
        expr=("log", ("var", 3)),
        anchor_node=None,
        scaffold_metadata={"form": "log_base"},
        local_fit_mse=0.1,
        local_probe_mse=0.1,
        local_mapping_kind="direct_linear_head",
        direct_metadata={"hole_node": ("var", 3), "feature_node": ("log", ("var", 3))},
    )
    call_log = []
    scored = []

    def _run_closure_search_pass_impl(**kwargs):
        proposal_context = kwargs.get("proposal_context")
        basis_state = getattr(proposal_context, "basis_state", None)
        call_log.append(
            len(tuple(getattr(basis_state, "blocks", ()) or ())) if basis_state is not None else 0
        )
        if len(call_log) == 1:
            return {
                "stats": {"scaffolds_enumerated": 2},
                "candidate_rows": [
                    {
                        "scaffold_id": "round1",
                        "expr": ("sqrt", ("var", 1)),
                        "child_key": "round1",
                        "local_probe_mse": 0.02,
                        "local_fit_mse": 0.02,
                        "candidate_child_size": 2,
                        "basis_state_obj": preview_state_round1,
                        "feature_block_obj": preview_state_round1.blocks[0],
                    },
                    {
                        "scaffold_id": "stale_same_round",
                        "expr": ("exp", ("var", 2)),
                        "child_key": "stale_same_round",
                        "local_probe_mse": 0.02,
                        "local_fit_mse": 0.02,
                        "candidate_child_size": 2,
                        "basis_state_obj": stale_same_round_state,
                        "feature_block_obj": stale_same_round_state.blocks[0],
                    },
                ],
            }
        return {
            "stats": {"scaffolds_enumerated": 1},
            "candidate_rows": [
                {
                    "scaffold_id": "round2",
                    "expr": ("log", ("var", 3)),
                    "child_key": "round2",
                    "local_probe_mse": 0.01,
                    "local_fit_mse": 0.01,
                    "candidate_child_size": 2,
                    "basis_state_obj": preview_state_round2,
                    "feature_block_obj": preview_state_round2.blocks[0],
                },
            ],
        }

    def _score_external_candidate_expr_fn(expr, **kwargs):
        scaffold_id = str(kwargs["candidate_meta"]["scaffold_id"])
        scored.append(scaffold_id)
        candidate_state = kwargs["candidate_meta"]["basis_state_obj"]
        raw_mse = 0.02 if scaffold_id == "round1" else 0.01
        return {
            "expr": expr,
            "raw_mse": raw_mse,
            "basis_state_obj": candidate_state,
        }

    kwargs = _base_closure_search_kwargs()
    kwargs.update(
        {
            "closure_search_exact_topk": 3,
            "proposal_context": ProposalContext(
                basis_state=base_state,
                basis_state_beam=admit_basis_state_to_beam([], base_state, beam_width=3),
                total_budget=3,
            ),
            "run_closure_search_pass_impl": _run_closure_search_pass_impl,
            "score_external_candidate_expr_fn": _score_external_candidate_expr_fn,
        }
    )

    run_closure_search_pass(**kwargs)

    stats = kwargs["closure_search_stats"]
    assert scored == ["round1", "stale_same_round", "round2"]
    assert len(call_log) >= 2
    assert int(call_log[1]) >= 2
    assert int(stats["basis_state_round_updates"]) == 1
    assert str(stats["basis_state_controller_mode"]) == "iterative_basis_loop"
    assert any(int(row.get("block_count", 0) or 0) >= 2 for row in list(stats["basis_state_beam"] or []))


def test_run_closure_search_pass_round_chooser_prefers_earlier_singleton_within_eff_mse_slack():
    base_state = basis_state_from_closure_candidate(
        family="affine",
        scaffold_id="base",
        expr=("var", 2),
        anchor_node=None,
        scaffold_metadata={"form": "base"},
        local_fit_mse=2.0,
        local_probe_mse=2.0,
        local_mapping_kind="direct_linear_head",
    )
    state_a = basis_state_from_closure_candidate(
        family="periodic",
        scaffold_id="a",
        expr=("cos", ("var", 0)),
        anchor_node=None,
        scaffold_metadata={"form": "cos_base"},
        local_fit_mse=0.11,
        local_probe_mse=0.11,
        local_mapping_kind="direct_linear_head",
    )
    state_b = basis_state_from_closure_candidate(
        family="power",
        scaffold_id="b",
        expr=("sqrt", ("var", 1)),
        anchor_node=None,
        scaffold_metadata={"form": "power_sqrt"},
        local_fit_mse=0.109,
        local_probe_mse=0.109,
        local_mapping_kind="direct_power_head",
    )
    scored = []

    def _run_closure_search_pass_impl(**kwargs):
        proposal_context = kwargs.get("proposal_context")
        basis_state = proposal_context.basis_state if isinstance(proposal_context, ProposalContext) else None
        if isinstance(basis_state, type(base_state)) and str(getattr(basis_state, "compiled_expr", None)) != str(base_state.compiled_expr):
            return {"stats": {}, "candidate_rows": []}
        return {
            "stats": {},
            "candidate_rows": [
                {
                    "scaffold_id": "a",
                    "expr": state_a.compiled_expr,
                    "child_key": "a",
                    "proposal_key": "a",
                    "local_probe_mse": 0.01,
                    "local_fit_mse": 0.01,
                    "candidate_child_size": 2,
                    "basis_state_obj": state_a,
                    "feature_block_obj": state_a.blocks[0],
                    "scaffold_family": "periodic",
                },
                {
                    "scaffold_id": "b",
                    "expr": state_b.compiled_expr,
                    "child_key": "b",
                    "proposal_key": "b",
                    "local_probe_mse": 0.02,
                    "local_fit_mse": 0.02,
                    "candidate_child_size": 2,
                    "basis_state_obj": state_b,
                    "feature_block_obj": state_b.blocks[0],
                    "scaffold_family": "power",
                },
            ],
        }

    def _score_external_candidate_expr_fn(expr, **kwargs):
        proposal_key = str(kwargs["candidate_meta"].get("proposal_key", ""))
        scored.append(proposal_key)
        candidate_state = kwargs["candidate_meta"]["basis_state_obj"]
        raw_mse = 0.11 if proposal_key == "a" else 0.109
        return {
            "expr": expr,
            "raw_mse": raw_mse,
            "fit_loss": raw_mse,
            "probe_loss": raw_mse,
            "basis_state_obj": candidate_state,
        }

    kwargs = _base_closure_search_kwargs()
    kwargs.update(
        {
            "closure_search_exact_topk": 2,
            "closure_search_pair_normal_enable": False,
            "closure_search_pair_rescue_enable": False,
            "closure_search_debug_topk": 4,
            "proposal_context": ProposalContext(
                basis_state=base_state,
                basis_state_beam=admit_basis_state_to_beam([], base_state, beam_width=4),
                total_budget=2,
            ),
            "run_closure_search_pass_impl": _run_closure_search_pass_impl,
            "score_external_candidate_expr_fn": _score_external_candidate_expr_fn,
            "score_native_candidate_basis_state_fn": None,
        }
    )

    run_closure_search_pass(**kwargs)

    stats = kwargs["closure_search_stats"]
    assert scored == ["a", "b"]
    assert int(stats["basis_state_round_commit_selected_singleton"]) == 1
    round_summaries = list(stats.get("debug_round_summaries", []) or [])
    assert round_summaries
    assert str(round_summaries[0].get("selected_commit_key", "")) == "a"


def test_run_closure_search_pass_empty_basis_seed_round_scores_multiple_rows_but_selects_one_commit():
    periodic_a = basis_state_from_closure_candidate(
        family="periodic",
        scaffold_id="periodic:cos:a",
        expr=("cos", ("var", 0)),
        anchor_node=None,
        scaffold_metadata={"form": "cos_base"},
        local_fit_mse=0.04,
        local_probe_mse=0.04,
        local_mapping_kind="direct_linear_head",
        direct_metadata={"feature_node": ("cos", ("var", 0)), "execution_mode": "exact_bound"},
    )
    periodic_b = basis_state_from_closure_candidate(
        family="periodic",
        scaffold_id="periodic:sin:b",
        expr=("sin", ("var", 0)),
        anchor_node=None,
        scaffold_metadata={"form": "sin_base"},
        local_fit_mse=0.05,
        local_probe_mse=0.05,
        local_mapping_kind="direct_linear_head",
        direct_metadata={"feature_node": ("sin", ("var", 0))},
    )
    power_state = basis_state_from_closure_candidate(
        family="power",
        scaffold_id="power:sqrt",
        expr=("sqrt", ("var", 1)),
        anchor_node=None,
        scaffold_metadata={"form": "power_sqrt"},
        local_fit_mse=0.06,
        local_probe_mse=0.06,
        local_mapping_kind="direct_power_head",
        direct_metadata={"power_inner_node": ("var", 1)},
    )
    call_log = []
    scored = []

    def _run_closure_search_pass_impl(**kwargs):
        proposal_context = kwargs.get("proposal_context")
        basis_state = getattr(proposal_context, "basis_state", None)
        call_log.append(
            len(tuple(getattr(basis_state, "blocks", ()) or ())) if basis_state is not None else 0
        )
        if len(call_log) == 1:
            return {
                "stats": {},
                "candidate_rows": [
                    {
                        "scaffold_id": "periodic:a",
                        "expr": ("cos", ("var", 0)),
                        "child_key": "periodic:a",
                        "proposal_key": "periodic:a",
                        "local_probe_mse": 0.04,
                        "local_fit_mse": 0.04,
                        "candidate_child_size": 2,
                        "basis_state_obj": periodic_a,
                        "feature_block_obj": periodic_a.blocks[0],
                        "scaffold_family": "periodic",
                        "execution_mode": "exact_bound",
                    },
                    {
                        "scaffold_id": "periodic:b",
                        "expr": ("sin", ("var", 0)),
                        "child_key": "periodic:b",
                        "proposal_key": "periodic:b",
                        "local_probe_mse": 0.05,
                        "local_fit_mse": 0.05,
                        "candidate_child_size": 2,
                        "basis_state_obj": periodic_b,
                        "feature_block_obj": periodic_b.blocks[0],
                        "scaffold_family": "periodic",
                    },
                    {
                        "scaffold_id": "power",
                        "expr": ("sqrt", ("var", 1)),
                        "child_key": "power",
                        "proposal_key": "power",
                        "local_probe_mse": 0.06,
                        "local_fit_mse": 0.06,
                        "candidate_child_size": 2,
                        "basis_state_obj": power_state,
                        "feature_block_obj": power_state.blocks[0],
                        "scaffold_family": "power",
                    },
                ],
            }
        return {"stats": {}, "candidate_rows": []}

    def _score_external_candidate_expr_fn(expr, **kwargs):
        scaffold_id = str(kwargs["candidate_meta"]["scaffold_id"])
        scored.append(scaffold_id)
        candidate_state = kwargs["candidate_meta"]["basis_state_obj"]
        raw_mse = float(kwargs["candidate_meta"].get("local_probe_mse", 1.0))
        return {
            "expr": expr,
            "raw_mse": raw_mse,
            "fit_loss": raw_mse,
            "probe_loss": raw_mse,
            "basis_state_obj": candidate_state,
        }

    kwargs = _base_closure_search_kwargs()
    kwargs.update(
        {
            "closure_search_exact_topk": 1,
            "closure_search_seed_exact_topk": 3,
            "closure_search_seed_beam_width": 2,
            "closure_search_seed_family_cap": 1,
            "run_closure_search_pass_impl": _run_closure_search_pass_impl,
            "score_external_candidate_expr_fn": _score_external_candidate_expr_fn,
        }
    )

    run_closure_search_pass(**kwargs)

    stats = kwargs["closure_search_stats"]
    assert len(scored) == 3
    assert scored[0] == "periodic:a"
    assert set(scored) == {"periodic:a", "periodic:b", "power"}
    assert len(call_log) == 2
    assert int(call_log[0]) == 0
    assert int(call_log[1]) >= 1
    assert bool(stats["basis_state_seed_mode_used"]) is True
    assert int(stats["basis_state_seed_rounds"]) == 1
    assert int(stats["basis_state_seed_scored"]) == 3
    assert int(stats["basis_state_round_commit_selected"]) == 1
    assert int(stats["basis_state_round_commit_selected_singleton"]) == 1
    assert int(stats["basis_state_round_commit_selected_pair"]) == 0
    assert int(stats["basis_state_beam_count"]) == 1
    beam_rows = list(stats["basis_state_beam"] or [])
    assert len(beam_rows) == 1
    assert tuple(beam_rows[0].get("block_families", []) or []) == ("periodic",)


def test_run_closure_search_pass_seed_exact_budget_is_independent_of_post_seed_exact_budget():
    state_a = basis_state_from_closure_candidate(
        family="periodic",
        scaffold_id="periodic:cos:a",
        expr=("cos", ("var", 0)),
        anchor_node=None,
        scaffold_metadata={"form": "cos_base"},
        local_fit_mse=0.04,
        local_probe_mse=0.04,
        local_mapping_kind="direct_linear_head",
        direct_metadata={"feature_node": ("cos", ("var", 0)), "execution_mode": "exact_bound"},
    )
    state_b = basis_state_from_closure_candidate(
        family="power",
        scaffold_id="power:sqrt:b",
        expr=("sqrt", ("var", 1)),
        anchor_node=None,
        scaffold_metadata={"form": "power_sqrt"},
        local_fit_mse=0.05,
        local_probe_mse=0.05,
        local_mapping_kind="direct_power_head",
        direct_metadata={"power_inner_node": ("var", 1)},
    )
    state_c = basis_state_from_closure_candidate(
        family="log",
        scaffold_id="log:base:c",
        expr=("log", ("var", 2)),
        anchor_node=None,
        scaffold_metadata={"form": "log_base"},
        local_fit_mse=0.06,
        local_probe_mse=0.06,
        local_mapping_kind="direct_linear_head",
        direct_metadata={"feature_node": ("log", ("var", 2))},
    )
    scored = []

    def _run_closure_search_pass_impl(**kwargs):
        proposal_context = kwargs.get("proposal_context")
        if getattr(proposal_context, "basis_state", None) is None:
            return {
                "stats": {},
                "candidate_rows": [
                    {
                        "scaffold_id": "a",
                        "expr": ("cos", ("var", 0)),
                        "child_key": "a",
                        "proposal_key": "a",
                        "local_probe_mse": 0.04,
                        "local_fit_mse": 0.04,
                        "candidate_child_size": 2,
                        "basis_state_obj": state_a,
                        "feature_block_obj": state_a.blocks[0],
                        "scaffold_family": "periodic",
                    },
                    {
                        "scaffold_id": "b",
                        "expr": ("sqrt", ("var", 1)),
                        "child_key": "b",
                        "proposal_key": "b",
                        "local_probe_mse": 0.05,
                        "local_fit_mse": 0.05,
                        "candidate_child_size": 2,
                        "basis_state_obj": state_b,
                        "feature_block_obj": state_b.blocks[0],
                        "scaffold_family": "power",
                    },
                    {
                        "scaffold_id": "c",
                        "expr": ("log", ("var", 2)),
                        "child_key": "c",
                        "proposal_key": "c",
                        "local_probe_mse": 0.06,
                        "local_fit_mse": 0.06,
                        "candidate_child_size": 2,
                        "basis_state_obj": state_c,
                        "feature_block_obj": state_c.blocks[0],
                        "scaffold_family": "log",
                    },
                ],
            }
        return {"stats": {}, "candidate_rows": []}

    def _score_external_candidate_expr_fn(expr, **kwargs):
        scored.append(str(kwargs["candidate_meta"]["scaffold_id"]))
        candidate_state = kwargs["candidate_meta"]["basis_state_obj"]
        raw_mse = float(kwargs["candidate_meta"].get("local_probe_mse", 1.0))
        return {
            "expr": expr,
            "raw_mse": raw_mse,
            "fit_loss": raw_mse,
            "probe_loss": raw_mse,
            "basis_state_obj": candidate_state,
        }

    kwargs = _base_closure_search_kwargs()
    kwargs.update(
        {
            "closure_search_exact_topk": 4,
            "closure_search_seed_exact_topk": 2,
            "closure_search_seed_beam_width": 2,
            "run_closure_search_pass_impl": _run_closure_search_pass_impl,
            "score_external_candidate_expr_fn": _score_external_candidate_expr_fn,
        }
    )

    run_closure_search_pass(**kwargs)

    stats = kwargs["closure_search_stats"]
    assert scored == ["a", "b"]
    assert bool(stats["basis_state_seed_mode_used"]) is True
    assert int(stats["basis_state_seed_scored"]) == 2


def test_run_closure_search_pass_seed_round_reserves_scaffold_budget_for_followup_rounds():
    seed_state = basis_state_from_closure_candidate(
        family="power",
        scaffold_id="power:sqr",
        expr=("sqr", ("var", 1)),
        anchor_node=None,
        scaffold_metadata={"form": "power_square"},
        local_fit_mse=0.04,
        local_probe_mse=0.04,
        local_mapping_kind="direct_power_head",
        direct_metadata={"power_inner_node": ("var", 1)},
    )
    call_budgets = []
    scored = []

    def _run_closure_search_pass_impl(**kwargs):
        call_budgets.append(int(kwargs.get("max_scaffolds", 0) or 0))
        if len(call_budgets) == 1:
            return {
                "stats": {"scaffolds_enumerated": int(kwargs.get("max_scaffolds", 0) or 0)},
                "candidate_rows": [
                    {
                        "scaffold_id": "seed",
                        "expr": ("sqr", ("var", 1)),
                        "child_key": "seed",
                        "proposal_key": "seed",
                        "local_probe_mse": 0.04,
                        "local_fit_mse": 0.04,
                        "candidate_child_size": 2,
                        "basis_state_obj": seed_state,
                        "feature_block_obj": seed_state.blocks[0],
                        "scaffold_family": "power",
                    }
                ],
            }
        return {
            "stats": {"scaffolds_enumerated": int(kwargs.get("max_scaffolds", 0) or 0)},
            "candidate_rows": [],
        }

    def _score_external_candidate_expr_fn(expr, **kwargs):
        scored.append(str(kwargs["candidate_meta"]["scaffold_id"]))
        candidate_state = kwargs["candidate_meta"]["basis_state_obj"]
        return {
            "expr": expr,
            "raw_mse": 0.04,
            "fit_loss": 0.04,
            "probe_loss": 0.04,
            "basis_state_obj": candidate_state,
        }

    kwargs = _base_closure_search_kwargs()
    kwargs.update(
        {
            "closure_search_families": ["periodic", "power", "quadratic", "rational"],
            "closure_search_max_proposals": 10,
            "closure_search_exact_topk": 2,
            "closure_search_seed_exact_topk": 1,
            "closure_search_seed_scaffold_reserve": 4,
            "run_closure_search_pass_impl": _run_closure_search_pass_impl,
            "score_external_candidate_expr_fn": _score_external_candidate_expr_fn,
        }
    )

    run_closure_search_pass(**kwargs)

    stats = kwargs["closure_search_stats"]
    assert scored == ["seed"]
    assert call_budgets[:2] == [6, 4]
    assert int(stats["basis_state_seed_scaffold_reserve"]) == 4
    assert int(stats["closure_search_rounds"]) >= 2


def test_run_closure_search_pass_aux_atoms_do_not_reserve_initial_core_scaffolds():
    call_budgets = []

    def _run_closure_search_pass_impl(**kwargs):
        call_budgets.append(int(kwargs.get("max_scaffolds", 0) or 0))
        return {
            "stats": {"scaffolds_enumerated": int(kwargs.get("max_scaffolds", 0) or 0)},
            "candidate_rows": [],
        }

    kwargs = _base_closure_search_kwargs()
    kwargs.update(
        {
            "closure_search_max_proposals": 48,
            "closure_search_exact_topk": 8,
            "closure_search_seed_scaffold_reserve": 0,
            "closure_search_emergent_aux_atoms_enable": True,
            "run_closure_search_pass_impl": _run_closure_search_pass_impl,
            "score_external_candidate_expr_fn": lambda expr, **_kwargs: {"expr": expr},
        }
    )

    run_closure_search_pass(**kwargs)

    stats = kwargs["closure_search_stats"]
    assert call_budgets == [48]
    assert int(stats["emergent_aux_atom_followup_reserved"]) == 0


def test_run_closure_search_pass_seed_exact_scoring_reserves_one_family_slot():
    periodic_state = basis_state_from_closure_candidate(
        family="periodic",
        scaffold_id="periodic:cos",
        expr=("cos", ("var", 0)),
        anchor_node=None,
        scaffold_metadata={"form": "cos_base"},
        local_fit_mse=0.01,
        local_probe_mse=0.01,
        local_mapping_kind="direct_linear_head",
        direct_metadata={"feature_node": ("cos", ("var", 0))},
    )
    rational_state = basis_state_from_closure_candidate(
        family="rational",
        scaffold_id="rational:affine",
        expr=("div", ("var", 0), ("add", ("var", 1), ("const", 1.0))),
        anchor_node=None,
        scaffold_metadata={"form": "rational_affine"},
        local_fit_mse=0.02,
        local_probe_mse=0.02,
        local_mapping_kind="direct_rational_head",
        direct_metadata={"u_node": ("var", 0), "v_node": ("var", 1)},
    )
    quadratic_state = basis_state_from_closure_candidate(
        family="quadratic",
        scaffold_id="quadratic:sqrt",
        expr=("sqrt", ("add", ("sqr", ("var", 0)), ("sqr", ("var", 1)))),
        anchor_node=None,
        scaffold_metadata={"form": "quadratic_sqrt"},
        local_fit_mse=0.03,
        local_probe_mse=0.03,
        local_mapping_kind="quadratic_sqrt",
        direct_metadata={"base_nodes": [("var", 0), ("var", 1)]},
    )
    power_state = basis_state_from_closure_candidate(
        family="power",
        scaffold_id="power:sqr",
        expr=("sqr", ("var", 2)),
        anchor_node=None,
        scaffold_metadata={"form": "power_square"},
        local_fit_mse=0.20,
        local_probe_mse=0.20,
        local_mapping_kind="direct_power_head",
        direct_metadata={"power_inner_node": ("var", 2)},
    )
    scored = []

    def _run_closure_search_pass_impl(**kwargs):
        proposal_context = kwargs.get("proposal_context")
        if getattr(proposal_context, "basis_state", None) is None:
            return {
                "stats": {},
                "candidate_rows": [
                    {
                        "scaffold_id": "periodic",
                        "expr": ("cos", ("var", 0)),
                        "child_key": "periodic",
                        "proposal_key": "periodic",
                        "local_probe_mse": 0.01,
                        "local_fit_mse": 0.01,
                        "candidate_child_size": 2,
                        "basis_state_obj": periodic_state,
                        "feature_block_obj": periodic_state.blocks[0],
                        "scaffold_family": "periodic",
                    },
                    {
                        "scaffold_id": "rational",
                        "expr": ("div", ("var", 0), ("add", ("var", 1), ("const", 1.0))),
                        "child_key": "rational",
                        "proposal_key": "rational",
                        "local_probe_mse": 0.02,
                        "local_fit_mse": 0.02,
                        "candidate_child_size": 5,
                        "basis_state_obj": rational_state,
                        "feature_block_obj": rational_state.blocks[0],
                        "scaffold_family": "rational",
                    },
                    {
                        "scaffold_id": "quadratic",
                        "expr": ("sqrt", ("add", ("sqr", ("var", 0)), ("sqr", ("var", 1)))),
                        "child_key": "quadratic",
                        "proposal_key": "quadratic",
                        "local_probe_mse": 0.03,
                        "local_fit_mse": 0.03,
                        "candidate_child_size": 6,
                        "basis_state_obj": quadratic_state,
                        "feature_block_obj": quadratic_state.blocks[0],
                        "scaffold_family": "quadratic",
                    },
                    {
                        "scaffold_id": "power",
                        "expr": ("sqr", ("var", 2)),
                        "child_key": "power",
                        "proposal_key": "power",
                        "local_probe_mse": 0.20,
                        "local_fit_mse": 0.20,
                        "candidate_child_size": 2,
                        "basis_state_obj": power_state,
                        "feature_block_obj": power_state.blocks[0],
                        "scaffold_family": "power",
                    },
                ],
            }
        return {"stats": {}, "candidate_rows": []}

    def _score_external_candidate_expr_fn(expr, **kwargs):
        scored.append(str(kwargs["candidate_meta"]["scaffold_id"]))
        candidate_state = kwargs["candidate_meta"]["basis_state_obj"]
        raw_mse = float(kwargs["candidate_meta"].get("local_probe_mse", 1.0))
        return {
            "expr": expr,
            "raw_mse": raw_mse,
            "fit_loss": raw_mse,
            "probe_loss": raw_mse,
            "basis_state_obj": candidate_state,
        }

    kwargs = _base_closure_search_kwargs()
    kwargs.update(
        {
            "closure_search_families": ["periodic", "rational", "quadratic", "power"],
            "closure_search_seed_exact_topk": 4,
            "closure_search_pair_rescue_enable": False,
            "run_closure_search_pass_impl": _run_closure_search_pass_impl,
            "score_external_candidate_expr_fn": _score_external_candidate_expr_fn,
            "score_native_candidate_basis_state_fn": None,
        }
    )

    run_closure_search_pass(**kwargs)

    stats = kwargs["closure_search_stats"]
    assert set(scored[:4]) == {"periodic", "rational", "quadratic", "power"}
    assert int(stats["basis_state_seed_family_reservations"]) >= 4


def test_run_closure_search_pass_beam_width_is_independent_of_exact_topk():
    state_a = basis_state_from_closure_candidate(
        family="periodic",
        scaffold_id="periodic:cos:a",
        expr=("cos", ("var", 0)),
        anchor_node=None,
        scaffold_metadata={"form": "cos_base"},
        local_fit_mse=0.04,
        local_probe_mse=0.04,
        local_mapping_kind="direct_linear_head",
        direct_metadata={"feature_node": ("cos", ("var", 0))},
    )
    state_b = basis_state_from_closure_candidate(
        family="power",
        scaffold_id="power:sqrt:b",
        expr=("sqrt", ("var", 1)),
        anchor_node=None,
        scaffold_metadata={"form": "power_sqrt"},
        local_fit_mse=0.05,
        local_probe_mse=0.05,
        local_mapping_kind="direct_power_head",
        direct_metadata={"power_inner_node": ("var", 1)},
    )
    state_c = basis_state_from_closure_candidate(
        family="log",
        scaffold_id="log:base:c",
        expr=("log", ("var", 2)),
        anchor_node=None,
        scaffold_metadata={"form": "log_base"},
        local_fit_mse=0.06,
        local_probe_mse=0.06,
        local_mapping_kind="direct_linear_head",
        direct_metadata={"feature_node": ("log", ("var", 2))},
    )
    initial_beam = admit_basis_state_to_beam([], state_a, beam_width=4)
    initial_beam = admit_basis_state_to_beam(initial_beam, state_b, beam_width=4)
    initial_beam = admit_basis_state_to_beam(initial_beam, state_c, beam_width=4)

    def _run_closure_search_pass_impl(**_kwargs):
        return {"stats": {}, "candidate_rows": []}

    kwargs = _base_closure_search_kwargs()
    kwargs.update(
        {
            "closure_search_exact_topk": 1,
            "closure_search_beam_width": 4,
            "proposal_context": ProposalContext(
                basis_state=state_a,
                basis_state_beam=initial_beam,
                total_budget=4,
            ),
            "run_closure_search_pass_impl": _run_closure_search_pass_impl,
            "score_external_candidate_expr_fn": lambda *args, **kwargs: None,
        }
    )

    run_closure_search_pass(**kwargs)

    stats = kwargs["closure_search_stats"]
    assert int(stats["basis_state_beam_width"]) == 4
    assert int(stats["basis_state_beam_count"]) == 3
    beam_rows = list(stats["basis_state_beam"] or [])
    beam_families = [tuple(row.get("block_families", []) or []) for row in beam_rows]
    assert ("periodic",) in beam_families
    assert ("power",) in beam_families
    assert ("log",) in beam_families


def test_run_closure_search_pass_state_aware_gain_ranks_complementary_block_over_better_local_probe():
    base_state = basis_state_from_closure_candidate(
        family="affine",
        scaffold_id="affine:x0",
        expr=("var", 0),
        anchor_node=None,
        scaffold_metadata={"form": "affine_base"},
        local_fit_mse=0.2,
        local_probe_mse=0.2,
        local_mapping_kind="direct_linear_head",
        direct_metadata={"feature_node": ("var", 0)},
    )
    redundant_state = basis_state_from_closure_candidate(
        family="affine",
        scaffold_id="affine:dup",
        expr=("add", ("var", 0), ("var", 0)),
        anchor_node=None,
        scaffold_metadata={"form": "affine_dup"},
        local_fit_mse=0.01,
        local_probe_mse=0.01,
        local_mapping_kind="direct_linear_head",
        direct_metadata={"feature_node": ("add", ("var", 0), ("var", 0))},
    )
    complementary_state = basis_state_from_closure_candidate(
        family="power",
        scaffold_id="power:sqrt",
        expr=("sqrt", ("var", 1)),
        anchor_node=None,
        scaffold_metadata={"form": "power_sqrt"},
        local_fit_mse=0.02,
        local_probe_mse=0.02,
        local_mapping_kind="direct_power_head",
        direct_metadata={"power_inner_node": ("var", 1)},
    )
    x = torch.tensor(
        [
            [0.2, 0.31],
            [0.4, 0.57],
            [0.6, 0.83],
            [0.8, 1.21],
            [1.0, 1.69],
            [1.2, 2.56],
        ],
        dtype=torch.float64,
    )
    y = x[:, :1] + torch.sqrt(x[:, 1:2])
    scored = []

    def _run_closure_search_pass_impl(**_kwargs):
        return {
            "stats": {},
            "candidate_rows": [
                {
                    "scaffold_id": "redundant",
                    "expr": ("add", ("var", 0), ("var", 0)),
                    "child_key": "redundant",
                    "proposal_key": "redundant",
                    "local_probe_mse": 1.0e-4,
                    "local_fit_mse": 1.0e-4,
                    "candidate_child_size": 3,
                    "basis_state_obj": redundant_state,
                    "feature_block_obj": redundant_state.blocks[0],
                    "scaffold_family": "affine",
                },
                {
                    "scaffold_id": "complementary",
                    "expr": ("sqrt", ("var", 1)),
                    "child_key": "complementary",
                    "proposal_key": "complementary",
                    "local_probe_mse": 2.0e-4,
                    "local_fit_mse": 2.0e-4,
                    "candidate_child_size": 2,
                    "basis_state_obj": complementary_state,
                    "feature_block_obj": complementary_state.blocks[0],
                    "scaffold_family": "power",
                },
            ],
        }

    def _score_external_candidate_expr_fn(expr, **kwargs):
        scaffold_id = str(kwargs["candidate_meta"]["scaffold_id"])
        scored.append(scaffold_id)
        candidate_state = kwargs["candidate_meta"]["basis_state_obj"]
        raw_mse = float(kwargs["candidate_meta"].get("local_probe_mse", 1.0))
        return {
            "expr": expr,
            "raw_mse": raw_mse,
            "fit_loss": raw_mse,
            "probe_loss": raw_mse,
            "basis_state_obj": candidate_state,
        }

    kwargs = _base_closure_search_kwargs()
    kwargs.update(
        {
            "closure_search_families": ["affine", "power"],
            "closure_search_exact_topk": 1,
            "x_fit": x,
            "y_fit": y,
            "x_probe": x,
            "y_probe": y,
            "proposal_context": ProposalContext(
                basis_state=base_state,
                basis_state_beam=admit_basis_state_to_beam([], base_state, beam_width=4),
                total_budget=2,
            ),
            "run_closure_search_pass_impl": _run_closure_search_pass_impl,
            "score_external_candidate_expr_fn": _score_external_candidate_expr_fn,
        }
    )

    run_closure_search_pass(**kwargs)

    stats = kwargs["closure_search_stats"]
    assert scored == ["complementary"]
    assert int(stats["basis_state_rank_proxy_candidates"]) >= 2
    assert int(stats["basis_state_rank_proxy_scored"]) >= 2


def test_run_closure_search_pass_exact_shortlist_reserves_family_diversity():
    base_state = basis_state_from_closure_candidate(
        family="affine",
        scaffold_id="affine:base",
        expr=("var", 3),
        anchor_node=None,
        scaffold_metadata={"form": "affine_base"},
        local_fit_mse=1.0,
        local_probe_mse=1.0,
        local_mapping_kind="direct_linear_head",
        direct_metadata={"feature_node": ("var", 3)},
    )
    periodic_a = basis_state_from_closure_candidate(
        family="periodic",
        scaffold_id="periodic:a",
        expr=("cos", ("var", 0)),
        anchor_node=None,
        scaffold_metadata={"form": "cos_base"},
        local_fit_mse=0.01,
        local_probe_mse=0.01,
        local_mapping_kind="direct_linear_head",
        direct_metadata={"feature_node": ("cos", ("var", 0))},
    )
    periodic_b = basis_state_from_closure_candidate(
        family="periodic",
        scaffold_id="periodic:b",
        expr=("sin", ("var", 0)),
        anchor_node=None,
        scaffold_metadata={"form": "sin_base"},
        local_fit_mse=0.02,
        local_probe_mse=0.02,
        local_mapping_kind="direct_linear_head",
        direct_metadata={"feature_node": ("sin", ("var", 0))},
    )
    power_state = basis_state_from_closure_candidate(
        family="power",
        scaffold_id="power:c",
        expr=("sqr", ("var", 1)),
        anchor_node=None,
        scaffold_metadata={"form": "power_square"},
        local_fit_mse=0.20,
        local_probe_mse=0.20,
        local_mapping_kind="direct_power_head",
        direct_metadata={"power_inner_node": ("var", 1)},
    )
    scored = []

    def _run_closure_search_pass_impl(**_kwargs):
        return {
            "stats": {},
            "candidate_rows": [
                {
                    "scaffold_id": "periodic:a",
                    "expr": ("cos", ("var", 0)),
                    "child_key": "periodic:a",
                    "proposal_key": "periodic:a",
                    "local_probe_mse": 0.01,
                    "local_fit_mse": 0.01,
                    "candidate_child_size": 2,
                    "basis_state_obj": periodic_a,
                    "feature_block_obj": periodic_a.blocks[0],
                    "scaffold_family": "periodic",
                },
                {
                    "scaffold_id": "periodic:b",
                    "expr": ("sin", ("var", 0)),
                    "child_key": "periodic:b",
                    "proposal_key": "periodic:b",
                    "local_probe_mse": 0.02,
                    "local_fit_mse": 0.02,
                    "candidate_child_size": 2,
                    "basis_state_obj": periodic_b,
                    "feature_block_obj": periodic_b.blocks[0],
                    "scaffold_family": "periodic",
                },
                {
                    "scaffold_id": "power:c",
                    "expr": ("sqr", ("var", 1)),
                    "child_key": "power:c",
                    "proposal_key": "power:c",
                    "local_probe_mse": 0.20,
                    "local_fit_mse": 0.20,
                    "candidate_child_size": 2,
                    "basis_state_obj": power_state,
                    "feature_block_obj": power_state.blocks[0],
                    "scaffold_family": "power",
                },
            ],
        }

    def _score_external_candidate_expr_fn(expr, **kwargs):
        scored.append(str(kwargs["candidate_meta"]["scaffold_id"]))
        candidate_state = kwargs["candidate_meta"]["basis_state_obj"]
        raw_mse = float(kwargs["candidate_meta"].get("local_probe_mse", 1.0))
        return {
            "expr": expr,
            "raw_mse": raw_mse,
            "fit_loss": raw_mse,
            "probe_loss": raw_mse,
            "basis_state_obj": candidate_state,
        }

    kwargs = _base_closure_search_kwargs()
    kwargs.update(
        {
            "closure_search_families": ["periodic", "power"],
            "closure_search_exact_topk": 2,
            "closure_search_pair_rescue_enable": False,
            "proposal_context": ProposalContext(
                basis_state=base_state,
                basis_state_beam=admit_basis_state_to_beam([], base_state, beam_width=4),
                total_budget=3,
            ),
            "run_closure_search_pass_impl": _run_closure_search_pass_impl,
            "score_external_candidate_expr_fn": _score_external_candidate_expr_fn,
            "score_native_candidate_basis_state_fn": None,
        }
    )

    run_closure_search_pass(**kwargs)

    stats = kwargs["closure_search_stats"]
    assert scored[:2] == ["periodic:a", "power:c"]
    assert int(stats["basis_state_family_reservations"]) >= 1


def test_run_closure_search_pass_exact_shortlist_preserves_same_signal_anchor_diversity():
    base_state = basis_state_from_closure_candidate(
        family="affine",
        scaffold_id="affine:base",
        expr=("var", 3),
        anchor_node=None,
        scaffold_metadata={"form": "affine_base"},
        local_fit_mse=1.0,
        local_probe_mse=1.0,
        local_mapping_kind="direct_linear_head",
        direct_metadata={"feature_node": ("var", 3)},
    )
    signal = ("cos", ("mul", ("var", 1), ("var", 2)))
    power_state = basis_state_from_closure_candidate(
        family="power",
        scaffold_id="power:full",
        expr=("mul", ("mul", ("var", 0), ("var", 3)), ("add", signal, ("sqr", signal))),
        anchor_node=("mul", ("var", 0), ("var", 3)),
        scaffold_metadata={"form": "power_square"},
        local_fit_mse=0.02,
        local_probe_mse=0.02,
        local_mapping_kind="direct_power_head",
        direct_metadata={
            "anchor_node": ("mul", ("var", 0), ("var", 3)),
            "hole_node": signal,
            "power_variant": "full_quadratic",
            "power_kind": "sqr_mul",
        },
    )
    periodic_same_anchor = basis_state_from_closure_candidate(
        family="periodic",
        scaffold_id="periodic:same_anchor",
        expr=("mul", ("mul", ("var", 0), ("var", 3)), signal),
        anchor_node=("mul", ("var", 0), ("var", 3)),
        scaffold_metadata={"form": "cos_mul"},
        local_fit_mse=0.01,
        local_probe_mse=0.01,
        local_mapping_kind="direct_harmonic_head",
        direct_metadata={
            "feature_node": signal,
            "harmonic_feature_nodes": [signal],
            "envelope_node": ("mul", ("var", 0), ("var", 3)),
        },
    )
    periodic_companion = basis_state_from_closure_candidate(
        family="periodic",
        scaffold_id="periodic:companion",
        expr=("mul", ("var", 0), signal),
        anchor_node=("var", 0),
        scaffold_metadata={"form": "cos_mul"},
        local_fit_mse=0.03,
        local_probe_mse=0.03,
        local_mapping_kind="direct_harmonic_head",
        direct_metadata={
            "feature_node": signal,
            "harmonic_feature_nodes": [signal],
            "envelope_node": ("var", 0),
        },
    )
    scored = []

    def _run_closure_search_pass_impl(**_kwargs):
        return {
            "stats": {},
            "candidate_rows": [
                {
                    "scaffold_id": "periodic:same_anchor",
                    "expr": ("mul", ("mul", ("var", 0), ("var", 3)), signal),
                    "child_key": "periodic:same_anchor",
                    "proposal_key": "periodic:same_anchor",
                    "local_probe_mse": 0.01,
                    "local_fit_mse": 0.01,
                    "candidate_child_size": 4,
                    "basis_state_obj": periodic_same_anchor,
                    "feature_block_obj": periodic_same_anchor.blocks[0],
                    "scaffold_family": "periodic",
                    "operator_id": "periodic:cos_mul",
                },
                {
                    "scaffold_id": "power:full",
                    "expr": ("mul", ("mul", ("var", 0), ("var", 3)), ("add", signal, ("sqr", signal))),
                    "child_key": "power:full",
                    "proposal_key": "power:full",
                    "local_probe_mse": 0.02,
                    "local_fit_mse": 0.02,
                    "candidate_child_size": 5,
                    "basis_state_obj": power_state,
                    "feature_block_obj": power_state.blocks[0],
                    "scaffold_family": "power",
                    "operator_id": "power:sqr_mul",
                    "direct_metadata": {
                        "anchor_node": ("mul", ("var", 0), ("var", 3)),
                        "hole_node": signal,
                        "power_variant": "full_quadratic",
                        "power_kind": "sqr_mul",
                    },
                },
                {
                    "scaffold_id": "periodic:companion",
                    "expr": ("mul", ("var", 0), signal),
                    "child_key": "periodic:companion",
                    "proposal_key": "periodic:companion",
                    "local_probe_mse": 0.03,
                    "local_fit_mse": 0.03,
                    "candidate_child_size": 3,
                    "basis_state_obj": periodic_companion,
                    "feature_block_obj": periodic_companion.blocks[0],
                    "scaffold_family": "periodic",
                    "operator_id": "periodic:cos_mul",
                },
            ],
        }

    def _score_external_candidate_expr_fn(expr, **kwargs):
        scored.append(str(kwargs["candidate_meta"]["scaffold_id"]))
        candidate_state = kwargs["candidate_meta"]["basis_state_obj"]
        raw_mse = float(kwargs["candidate_meta"].get("local_probe_mse", 1.0))
        return {
            "expr": expr,
            "raw_mse": raw_mse,
            "fit_loss": raw_mse,
            "probe_loss": raw_mse,
            "basis_state_obj": candidate_state,
        }

    kwargs = _base_closure_search_kwargs()
    kwargs.update(
        {
            "closure_search_families": ["periodic", "power"],
            "closure_search_exact_topk": 3,
            "closure_search_pair_rescue_enable": False,
            "proposal_context": ProposalContext(
                basis_state=base_state,
                basis_state_beam=admit_basis_state_to_beam([], base_state, beam_width=4),
                total_budget=3,
            ),
            "run_closure_search_pass_impl": _run_closure_search_pass_impl,
            "score_external_candidate_expr_fn": _score_external_candidate_expr_fn,
            "score_native_candidate_basis_state_fn": None,
        }
    )

    run_closure_search_pass(**kwargs)

    stats = kwargs["closure_search_stats"]
    assert scored[:3] == ["periodic:same_anchor", "power:full", "periodic:companion"]
    assert int(stats["basis_state_interaction_anchor_reservations"]) >= 1


def test_run_closure_search_pass_seed_exact_shortlist_prioritizes_same_signal_companion_before_trailing_seed_families():
    signal = ("cos", ("mul", ("var", 1), ("var", 2)))
    power_full = basis_state_from_closure_candidate(
        family="power",
        scaffold_id="power:full",
        expr=("mul", ("mul", ("var", 0), ("var", 3)), ("add", signal, ("sqr", signal))),
        anchor_node=("mul", ("var", 0), ("var", 3)),
        scaffold_metadata={"form": "power_square"},
        local_fit_mse=0.23,
        local_probe_mse=0.23,
        local_mapping_kind="direct_power_head",
        direct_metadata={
            "anchor_node": ("mul", ("var", 0), ("var", 3)),
            "hole_node": signal,
            "power_variant": "full_quadratic",
            "power_kind": "sqr_mul",
        },
    )
    power_square = basis_state_from_closure_candidate(
        family="power",
        scaffold_id="power:square",
        expr=("mul", ("mul", ("var", 0), ("var", 3)), ("sqr", signal)),
        anchor_node=("mul", ("var", 0), ("var", 3)),
        scaffold_metadata={"form": "power_square"},
        local_fit_mse=1.93,
        local_probe_mse=1.93,
        local_mapping_kind="direct_power_head",
        direct_metadata={
            "anchor_node": ("mul", ("var", 0), ("var", 3)),
            "hole_node": signal,
            "power_variant": "square_only",
            "power_kind": "sqr_mul",
        },
    )
    rational_state = basis_state_from_closure_candidate(
        family="rational",
        scaffold_id="rational:demo",
        expr=("div", ("mul", ("var", 0), signal), ("add", 1, ("cos", ("var", 3)))),
        anchor_node=None,
        scaffold_metadata={"form": "rational_demo"},
        local_fit_mse=2.08,
        local_probe_mse=2.08,
        local_mapping_kind="direct_rational_head",
        direct_metadata={"u_node": ("mul", ("var", 0), signal), "v_node": ("cos", ("var", 3))},
    )
    periodic_wrong = basis_state_from_closure_candidate(
        family="periodic",
        scaffold_id="periodic:wrong",
        expr=("sub", ("mul", ("var", 0), ("var", 3)), ("var", 0)),
        anchor_node=("mul", ("mul", ("var", 1), ("var", 2)), ("var", 0)),
        scaffold_metadata={"form": "cos_mul"},
        local_fit_mse=2.50,
        local_probe_mse=2.50,
        local_mapping_kind="direct_harmonic_head",
        direct_metadata={
            "feature_node": ("cos", ("cos", ("var", 3))),
            "harmonic_feature_nodes": [("cos", ("cos", ("var", 3))), ("sin", ("cos", ("var", 3)))],
            "envelope_node": ("mul", ("mul", ("var", 1), ("var", 2)), ("var", 0)),
            "companion_nodes": [("var", 0), ("mul", ("var", 0), ("var", 3))],
        },
    )
    periodic_companion = basis_state_from_closure_candidate(
        family="periodic",
        scaffold_id="periodic:companion",
        expr=("mul", ("var", 0), signal),
        anchor_node=("var", 0),
        scaffold_metadata={"form": "cos_mul"},
        local_fit_mse=2.96,
        local_probe_mse=2.96,
        local_mapping_kind="direct_harmonic_head",
        direct_metadata={
            "feature_node": signal,
            "harmonic_feature_nodes": [("cos", ("mul", ("var", 1), ("var", 2))), ("sin", ("mul", ("var", 1), ("var", 2)))],
            "envelope_node": ("var", 0),
        },
    )
    quadratic_state = basis_state_from_closure_candidate(
        family="quadratic",
        scaffold_id="quadratic:demo",
        expr=("mul", ("sqrt", ("add", ("sqr", ("mul", ("var", 1), ("var", 2))), ("sqr", ("var", 3)))), ("var", 0)),
        anchor_node=("var", 0),
        scaffold_metadata={"form": "sqrt_mul"},
        local_fit_mse=10.81,
        local_probe_mse=10.81,
        local_mapping_kind="direct_quadratic_head",
        direct_metadata={"anchor_node": ("var", 0), "quadratic_latent_node": ("sqrt", ("add", ("sqr", ("mul", ("var", 1), ("var", 2))), ("sqr", ("var", 3))))},
    )
    exp_state = basis_state_from_closure_candidate(
        family="exp",
        scaffold_id="exp:demo",
        expr=("mul", ("exp", ("var", 3)), ("var", 0)),
        anchor_node=("var", 0),
        scaffold_metadata={"form": "exp_mul"},
        local_fit_mse=3.40,
        local_probe_mse=3.40,
        local_mapping_kind="direct_exp_mul_head",
        direct_metadata={"feature_node": ("exp", ("var", 3)), "anchor_node": ("var", 0)},
    )
    scored = []

    def _run_closure_search_pass_impl(**_kwargs):
        return {
            "stats": {},
            "candidate_rows": [
                {
                    "scaffold_id": "power:full",
                    "expr": power_full.compiled_expr,
                    "child_key": "power:full",
                    "proposal_key": "power:full",
                    "local_probe_mse": 0.23,
                    "local_fit_mse": 0.23,
                    "candidate_child_size": 5,
                    "basis_state_obj": power_full,
                    "feature_block_obj": power_full.blocks[0],
                    "scaffold_family": "power",
                    "operator_id": "power:sqr_mul",
                },
                {
                    "scaffold_id": "rational:demo",
                    "expr": rational_state.compiled_expr,
                    "child_key": "rational:demo",
                    "proposal_key": "rational:demo",
                    "local_probe_mse": 2.08,
                    "local_fit_mse": 2.08,
                    "candidate_child_size": 6,
                    "basis_state_obj": rational_state,
                    "feature_block_obj": rational_state.blocks[0],
                    "scaffold_family": "rational",
                    "operator_id": "rational:affine",
                },
                {
                    "scaffold_id": "periodic:wrong",
                    "expr": periodic_wrong.compiled_expr,
                    "child_key": "periodic:wrong",
                    "proposal_key": "periodic:wrong",
                    "local_probe_mse": 2.50,
                    "local_fit_mse": 2.50,
                    "candidate_child_size": 5,
                    "basis_state_obj": periodic_wrong,
                    "feature_block_obj": periodic_wrong.blocks[0],
                    "scaffold_family": "periodic",
                    "operator_id": "periodic:cos_mul",
                },
                {
                    "scaffold_id": "quadratic:demo",
                    "expr": quadratic_state.compiled_expr,
                    "child_key": "quadratic:demo",
                    "proposal_key": "quadratic:demo",
                    "local_probe_mse": 10.81,
                    "local_fit_mse": 10.81,
                    "candidate_child_size": 6,
                    "basis_state_obj": quadratic_state,
                    "feature_block_obj": quadratic_state.blocks[0],
                    "scaffold_family": "quadratic",
                    "operator_id": "quadratic:sqrt_mul",
                },
                {
                    "scaffold_id": "exp:demo",
                    "expr": exp_state.compiled_expr,
                    "child_key": "exp:demo",
                    "proposal_key": "exp:demo",
                    "local_probe_mse": 3.40,
                    "local_fit_mse": 3.40,
                    "candidate_child_size": 4,
                    "basis_state_obj": exp_state,
                    "feature_block_obj": exp_state.blocks[0],
                    "scaffold_family": "exp",
                    "operator_id": "exp:mul",
                },
                {
                    "scaffold_id": "power:square",
                    "expr": power_square.compiled_expr,
                    "child_key": "power:square",
                    "proposal_key": "power:square",
                    "local_probe_mse": 1.93,
                    "local_fit_mse": 1.93,
                    "candidate_child_size": 5,
                    "basis_state_obj": power_square,
                    "feature_block_obj": power_square.blocks[0],
                    "scaffold_family": "power",
                    "operator_id": "power:sqr_mul",
                    "local_mapping_nparams": 1,
                },
                {
                    "scaffold_id": "periodic:companion",
                    "expr": periodic_companion.compiled_expr,
                    "child_key": "periodic:companion",
                    "proposal_key": "periodic:companion",
                    "local_probe_mse": 2.96,
                    "local_fit_mse": 2.96,
                    "candidate_child_size": 3,
                    "basis_state_obj": periodic_companion,
                    "feature_block_obj": periodic_companion.blocks[0],
                    "scaffold_family": "periodic",
                    "operator_id": "periodic:cos_mul",
                },
            ],
        }

    def _score_external_candidate_expr_fn(expr, **kwargs):
        scored.append(str(kwargs["candidate_meta"]["scaffold_id"]))
        candidate_state = kwargs["candidate_meta"]["basis_state_obj"]
        raw_mse = float(kwargs["candidate_meta"].get("local_probe_mse", 1.0))
        return {
            "expr": expr,
            "raw_mse": raw_mse,
            "fit_loss": raw_mse,
            "probe_loss": raw_mse,
            "basis_state_obj": candidate_state,
        }

    kwargs = _base_closure_search_kwargs()
    kwargs.update(
        {
            "closure_search_families": ["periodic", "power", "rational", "quadratic", "exp"],
            "closure_search_exact_topk": 2,
            "closure_search_seed_exact_topk": 6,
            "closure_search_pair_rescue_enable": False,
            "run_closure_search_pass_impl": _run_closure_search_pass_impl,
            "score_external_candidate_expr_fn": _score_external_candidate_expr_fn,
            "score_native_candidate_basis_state_fn": None,
        }
    )

    run_closure_search_pass(**kwargs)

    stats = kwargs["closure_search_stats"]
    first_six = scored[:6]
    assert first_six[0] == "power:full"
    assert "periodic:companion" in first_six
    assert "power:square" in first_six
    assert first_six.index("power:full") < first_six.index("power:square")
    assert int(stats["basis_state_interaction_anchor_reservations"]) >= 1


def test_run_closure_search_pass_seed_precommit_pair_adds_same_signal_joint_state_to_beam():
    signal = ("cos", ("mul", ("var", 1), ("var", 2)))
    power_state = basis_state_from_closure_candidate(
        family="power",
        scaffold_id="power:full",
        expr=("mul", ("mul", ("var", 0), ("var", 3)), ("add", signal, ("sqr", signal))),
        anchor_node=("mul", ("var", 0), ("var", 3)),
        scaffold_metadata={"form": "power_square"},
        local_fit_mse=0.23,
        local_probe_mse=0.23,
        local_mapping_kind="direct_power_head",
        direct_metadata={
            "anchor_node": ("mul", ("var", 0), ("var", 3)),
            "hole_node": signal,
            "power_variant": "full_quadratic",
            "power_kind": "sqr_mul",
        },
    )
    periodic_state = basis_state_from_closure_candidate(
        family="periodic",
        scaffold_id="periodic:companion",
        expr=("mul", ("var", 0), signal),
        anchor_node=("var", 0),
        scaffold_metadata={"form": "cos_mul"},
        local_fit_mse=2.96,
        local_probe_mse=2.96,
        local_mapping_kind="direct_harmonic_head",
        direct_metadata={
            "feature_node": signal,
            "harmonic_feature_nodes": [("cos", ("mul", ("var", 1), ("var", 2))), ("sin", ("mul", ("var", 1), ("var", 2)))],
            "envelope_node": ("var", 0),
        },
    )
    distractor_state = basis_state_from_closure_candidate(
        family="rational",
        scaffold_id="rational:demo",
        expr=("div", ("mul", ("var", 0), signal), ("add", 1, ("cos", ("var", 3)))),
        anchor_node=None,
        scaffold_metadata={"form": "rational_demo"},
        local_fit_mse=2.08,
        local_probe_mse=2.08,
        local_mapping_kind="direct_rational_head",
        direct_metadata={"u_node": ("mul", ("var", 0), signal), "v_node": ("cos", ("var", 3))},
    )
    scored = []

    def _run_closure_search_pass_impl(**_kwargs):
        return {
            "stats": {},
            "candidate_rows": [
                {
                    "scaffold_id": "power:full",
                    "expr": power_state.compiled_expr,
                    "child_key": "power:full",
                    "proposal_key": "power:full",
                    "local_probe_mse": 0.23,
                    "local_fit_mse": 0.23,
                    "candidate_child_size": 5,
                    "basis_state_obj": power_state,
                    "feature_block_obj": power_state.blocks[0],
                    "scaffold_family": "power",
                    "operator_id": "power:sqr_mul",
                },
                {
                    "scaffold_id": "rational:demo",
                    "expr": distractor_state.compiled_expr,
                    "child_key": "rational:demo",
                    "proposal_key": "rational:demo",
                    "local_probe_mse": 2.08,
                    "local_fit_mse": 2.08,
                    "candidate_child_size": 6,
                    "basis_state_obj": distractor_state,
                    "feature_block_obj": distractor_state.blocks[0],
                    "scaffold_family": "rational",
                    "operator_id": "rational:affine",
                },
                {
                    "scaffold_id": "periodic:companion",
                    "expr": periodic_state.compiled_expr,
                    "child_key": "periodic:companion",
                    "proposal_key": "periodic:companion",
                    "local_probe_mse": 2.96,
                    "local_fit_mse": 2.96,
                    "candidate_child_size": 3,
                    "basis_state_obj": periodic_state,
                    "feature_block_obj": periodic_state.blocks[0],
                    "scaffold_family": "periodic",
                    "operator_id": "periodic:cos_mul",
                },
            ],
        }

    def _score_external_candidate_expr_fn(expr, **kwargs):
        scored.append(str(kwargs["candidate_meta"]["scaffold_id"]))
        candidate_state = kwargs["candidate_meta"]["basis_state_obj"]
        proposal_key = str(kwargs["candidate_meta"].get("proposal_key", ""))
        raw_mse = 1.0e-6 if proposal_key.startswith("pair_precommit::") else float(
            kwargs["candidate_meta"].get("local_probe_mse", 1.0)
        )
        return {
            "expr": expr,
            "raw_mse": raw_mse,
            "fit_loss": raw_mse,
            "probe_loss": raw_mse,
            "basis_state_obj": candidate_state,
        }

    kwargs = _base_closure_search_kwargs()
    kwargs.update(
        {
            "closure_search_families": ["power", "rational", "periodic"],
            "closure_search_seed_exact_topk": 3,
            "closure_search_seed_beam_width": 4,
            "closure_search_pair_rescue_enable": True,
            "closure_search_pair_rescue_topk": 3,
            "closure_search_pair_rescue_max_pairs": 2,
            "run_closure_search_pass_impl": _run_closure_search_pass_impl,
            "score_external_candidate_expr_fn": _score_external_candidate_expr_fn,
            "score_native_candidate_basis_state_fn": None,
        }
    )

    run_closure_search_pass(**kwargs)

    stats = kwargs["closure_search_stats"]
    assert int(stats["basis_state_pair_precommit_scored"]) >= 1
    assert int(stats["basis_state_pair_precommit_accepted"]) >= 1
    assert int(stats["basis_state_round_commit_selected_pair"]) == 1
    assert any(str(token).startswith("pair_precommit::") for token in scored)
    accepted_pairs = list(stats.get("accepted_pair_events", []) or [])
    selected_pairs = list(stats.get("selected_pair_events", []) or [])
    assert accepted_pairs
    assert selected_pairs
    assert str(accepted_pairs[0].get("profile", "")) == "seed_precommit"
    assert str(selected_pairs[0].get("profile", "")) == "seed_precommit"
    assert "same_interaction" in list(accepted_pairs[0].get("relation_tags", []) or [])
    assert all(
        str(source) == "exact_scored_singleton"
        for source in list(accepted_pairs[0].get("pair_member_sources", []) or [])
    )
    beam_rows = list(stats["basis_state_beam"] or [])
    assert any(int(row.get("block_count", 0) or 0) >= 2 for row in beam_rows)


def test_run_closure_search_pass_exact_shortlist_keeps_distinct_power_variants():
    periodic_state = basis_state_from_closure_candidate(
        family="periodic",
        scaffold_id="periodic:a",
        expr=("cos", ("var", 0)),
        anchor_node=None,
        scaffold_metadata={"form": "cos_base"},
        local_fit_mse=0.01,
        local_probe_mse=0.01,
        local_mapping_kind="direct_linear_head",
        direct_metadata={"feature_node": ("cos", ("var", 0))},
    )
    periodic_extra_state = basis_state_from_closure_candidate(
        family="periodic",
        scaffold_id="periodic:b",
        expr=("sin", ("var", 0)),
        anchor_node=None,
        scaffold_metadata={"form": "sin_base"},
        local_fit_mse=0.021,
        local_probe_mse=0.021,
        local_mapping_kind="direct_linear_head",
        direct_metadata={"feature_node": ("sin", ("var", 0))},
    )
    power_full_state = basis_state_from_closure_candidate(
        family="power",
        scaffold_id="power:full",
        expr=("add", ("var", 0), ("sqr", ("var", 1))),
        anchor_node=("var", 0),
        scaffold_metadata={"form": "power_square"},
        local_fit_mse=0.02,
        local_probe_mse=0.02,
        local_mapping_kind="direct_power_head",
        direct_metadata={
            "anchor_node": ("var", 0),
            "hole_node": ("cos", ("mul", ("var", 1), ("var", 2))),
            "power_variant": "full_quadratic",
            "power_kind": "sqr_mul",
        },
    )
    power_square_state = basis_state_from_closure_candidate(
        family="power",
        scaffold_id="power:square",
        expr=("sqr", ("var", 1)),
        anchor_node=("var", 0),
        scaffold_metadata={"form": "power_square"},
        local_fit_mse=0.03,
        local_probe_mse=0.03,
        local_mapping_kind="direct_power_head",
        direct_metadata={
            "anchor_node": ("var", 0),
            "hole_node": ("cos", ("mul", ("var", 1), ("var", 2))),
            "power_variant": "square_only",
            "power_kind": "sqr_mul",
        },
    )
    power_linear_state = basis_state_from_closure_candidate(
        family="power",
        scaffold_id="power:linear",
        expr=("add", ("var", 1), ("sqr", ("var", 1))),
        anchor_node=("var", 0),
        scaffold_metadata={"form": "power_square"},
        local_fit_mse=0.025,
        local_probe_mse=0.025,
        local_mapping_kind="direct_power_head",
        direct_metadata={
            "anchor_node": ("var", 0),
            "hole_node": ("cos", ("mul", ("var", 1), ("var", 2))),
            "power_variant": "linear_square",
            "power_kind": "sqr_mul",
        },
    )
    scored = []

    def _run_closure_search_pass_impl(**_kwargs):
        return {
            "stats": {},
            "candidate_rows": [
                {
                    "scaffold_id": "periodic:a",
                    "expr": ("cos", ("var", 0)),
                    "child_key": "periodic:a",
                    "proposal_key": "periodic:a",
                    "local_probe_mse": 0.01,
                    "local_fit_mse": 0.01,
                    "candidate_child_size": 2,
                    "basis_state_obj": periodic_state,
                    "feature_block_obj": periodic_state.blocks[0],
                    "scaffold_family": "periodic",
                },
                {
                    "scaffold_id": "power:full",
                    "expr": ("add", ("var", 0), ("sqr", ("var", 1))),
                    "child_key": "power:full",
                    "proposal_key": "power:full",
                    "local_probe_mse": 0.02,
                    "local_fit_mse": 0.02,
                    "candidate_child_size": 3,
                    "basis_state_obj": power_full_state,
                    "feature_block_obj": power_full_state.blocks[0],
                    "scaffold_family": "power",
                    "operator_id": "power:sqr_mul",
                    "direct_metadata": {
                        "anchor_node": ("var", 0),
                        "hole_node": ("cos", ("mul", ("var", 1), ("var", 2))),
                        "power_variant": "full_quadratic",
                        "power_kind": "sqr_mul",
                    },
                },
                {
                    "scaffold_id": "periodic:b",
                    "expr": ("sin", ("var", 0)),
                    "child_key": "periodic:b",
                    "proposal_key": "periodic:b",
                    "local_probe_mse": 0.021,
                    "local_fit_mse": 0.021,
                    "candidate_child_size": 2,
                    "basis_state_obj": periodic_extra_state,
                    "feature_block_obj": periodic_extra_state.blocks[0],
                    "scaffold_family": "periodic",
                },
                {
                    "scaffold_id": "power:linear",
                    "expr": ("add", ("var", 1), ("sqr", ("var", 1))),
                    "child_key": "power:linear",
                    "proposal_key": "power:linear",
                    "local_probe_mse": 0.025,
                    "local_fit_mse": 0.025,
                    "candidate_child_size": 3,
                    "basis_state_obj": power_linear_state,
                    "feature_block_obj": power_linear_state.blocks[0],
                    "scaffold_family": "power",
                    "operator_id": "power:sqr_mul",
                    "local_mapping_nparams": 2,
                    "direct_metadata": {
                        "anchor_node": ("var", 0),
                        "hole_node": ("cos", ("mul", ("var", 1), ("var", 2))),
                        "power_variant": "linear_square",
                        "power_kind": "sqr_mul",
                    },
                },
                {
                    "scaffold_id": "power:square",
                    "expr": ("sqr", ("var", 1)),
                    "child_key": "power:square",
                    "proposal_key": "power:square",
                    "local_probe_mse": 0.03,
                    "local_fit_mse": 0.03,
                    "candidate_child_size": 2,
                    "basis_state_obj": power_square_state,
                    "feature_block_obj": power_square_state.blocks[0],
                    "scaffold_family": "power",
                    "operator_id": "power:sqr_mul",
                    "local_mapping_nparams": 1,
                    "direct_metadata": {
                        "anchor_node": ("var", 0),
                        "hole_node": ("cos", ("mul", ("var", 1), ("var", 2))),
                        "power_variant": "square_only",
                        "power_kind": "sqr_mul",
                    },
                },
            ],
        }

    def _score_external_candidate_expr_fn(expr, **kwargs):
        scored.append(str(kwargs["candidate_meta"]["scaffold_id"]))
        candidate_state = kwargs["candidate_meta"]["basis_state_obj"]
        raw_mse = float(kwargs["candidate_meta"].get("local_probe_mse", 1.0))
        return {
            "expr": expr,
            "raw_mse": raw_mse,
            "fit_loss": raw_mse,
            "probe_loss": raw_mse,
            "basis_state_obj": candidate_state,
        }

    kwargs = _base_closure_search_kwargs()
    kwargs.update(
        {
            "closure_search_families": ["periodic", "power"],
            "closure_search_seed_exact_topk": 3,
            "closure_search_pair_rescue_enable": False,
            "run_closure_search_pass_impl": _run_closure_search_pass_impl,
            "score_external_candidate_expr_fn": _score_external_candidate_expr_fn,
            "score_native_candidate_basis_state_fn": None,
        }
    )

    run_closure_search_pass(**kwargs)

    stats = kwargs["closure_search_stats"]
    assert scored[:3] == ["periodic:a", "power:full", "power:square"]
    assert int(stats["basis_state_variant_reservations"]) >= 1


def test_run_closure_search_pass_explores_unexpanded_seed_beam_states_before_stopping():
    periodic_state = basis_state_from_closure_candidate(
        family="periodic",
        scaffold_id="periodic:a",
        expr=("cos", ("var", 0)),
        anchor_node=None,
        scaffold_metadata={"form": "cos_base"},
        local_fit_mse=0.01,
        local_probe_mse=0.01,
        local_mapping_kind="direct_linear_head",
        direct_metadata={"feature_node": ("cos", ("var", 0))},
    )
    power_state = basis_state_from_closure_candidate(
        family="power",
        scaffold_id="power:b",
        expr=("sqr", ("var", 1)),
        anchor_node=None,
        scaffold_metadata={"form": "power_square"},
        local_fit_mse=0.02,
        local_probe_mse=0.02,
        local_mapping_kind="direct_power_head",
        direct_metadata={"power_inner_node": ("var", 1)},
    )
    initial_beam = admit_basis_state_to_beam([], periodic_state, beam_width=2)
    initial_beam = admit_basis_state_to_beam(initial_beam, power_state, beam_width=2)
    seen_round_basis = []

    def _run_closure_search_pass_impl(**kwargs):
        proposal_context = kwargs.get("proposal_context")
        basis_state = proposal_context.basis_state if isinstance(proposal_context, ProposalContext) else None
        seen_round_basis.append(
            str(basis_state.compiled_expr) if isinstance(basis_state, type(periodic_state)) else ""
        )
        if isinstance(basis_state, type(periodic_state)):
            family = basis_state.blocks[0].family
            expr = basis_state.compiled_expr
            return {
                "stats": {},
                "candidate_rows": [
                    {
                        "scaffold_id": f"{family}:noop",
                        "expr": expr,
                        "child_key": f"{family}:noop",
                        "proposal_key": f"{family}:noop",
                        "local_probe_mse": 1.0,
                        "local_fit_mse": 1.0,
                        "candidate_child_size": 2,
                        "basis_state_obj": basis_state,
                        "feature_block_obj": basis_state.blocks[0],
                        "scaffold_family": family,
                    }
                ],
            }
        return {"stats": {}, "candidate_rows": []}

    def _score_external_candidate_expr_fn(expr, **kwargs):
        candidate_state = kwargs["candidate_meta"]["basis_state_obj"]
        raw_mse = float(kwargs["candidate_meta"].get("local_probe_mse", 1.0))
        return {
            "expr": expr,
            "raw_mse": raw_mse,
            "fit_loss": raw_mse,
            "probe_loss": raw_mse,
            "basis_state_obj": candidate_state,
        }

    kwargs = _base_closure_search_kwargs()
    kwargs.update(
        {
            "closure_search_families": ["periodic", "power"],
            "closure_search_max_proposals": 16,
            "closure_search_exact_topk": 2,
            "closure_search_seed_exact_topk": 2,
            "closure_search_seed_beam_width": 2,
            "closure_search_beam_width": 2,
            "closure_search_pair_rescue_enable": False,
            "proposal_context": ProposalContext(
                basis_state=periodic_state,
                basis_state_beam=initial_beam,
                total_budget=16,
            ),
            "run_closure_search_pass_impl": _run_closure_search_pass_impl,
            "score_external_candidate_expr_fn": _score_external_candidate_expr_fn,
            "score_native_candidate_basis_state_fn": None,
        }
    )

    run_closure_search_pass(**kwargs)

    assert len(seen_round_basis) == 2
    assert seen_round_basis[0] != seen_round_basis[1]


def test_run_closure_search_pass_uses_nonseed_exact_budget_per_round():
    periodic_state = basis_state_from_closure_candidate(
        family="periodic",
        scaffold_id="periodic:seed",
        expr=("cos", ("var", 0)),
        anchor_node=None,
        scaffold_metadata={"form": "cos_base"},
        local_fit_mse=0.01,
        local_probe_mse=0.01,
        local_mapping_kind="direct_linear_head",
        direct_metadata={"feature_node": ("cos", ("var", 0))},
    )
    power_state = basis_state_from_closure_candidate(
        family="power",
        scaffold_id="power:seed",
        expr=("sqr", ("var", 1)),
        anchor_node=None,
        scaffold_metadata={"form": "power_square"},
        local_fit_mse=0.02,
        local_probe_mse=0.02,
        local_mapping_kind="direct_power_head",
        direct_metadata={"power_inner_node": ("var", 1)},
    )
    seen_round_basis = []

    def _run_closure_search_pass_impl(**kwargs):
        proposal_context = kwargs.get("proposal_context")
        basis_state = proposal_context.basis_state if isinstance(proposal_context, ProposalContext) else None
        seen_round_basis.append(
            str(basis_state.compiled_expr) if isinstance(basis_state, type(periodic_state)) else ""
        )
        if not isinstance(basis_state, type(periodic_state)):
            return {
                "stats": {},
                "candidate_rows": [
                    {
                        "scaffold_id": "periodic:seed",
                        "expr": periodic_state.compiled_expr,
                        "child_key": "periodic:seed",
                        "proposal_key": "periodic:seed",
                        "local_probe_mse": 0.01,
                        "local_fit_mse": 0.01,
                        "candidate_child_size": 2,
                        "basis_state_obj": periodic_state,
                        "basis_state_dict": periodic_state.to_dict(),
                        "scaffold_family": "periodic",
                    },
                    {
                        "scaffold_id": "power:seed",
                        "expr": power_state.compiled_expr,
                        "child_key": "power:seed",
                        "proposal_key": "power:seed",
                        "local_probe_mse": 0.02,
                        "local_fit_mse": 0.02,
                        "candidate_child_size": 2,
                        "basis_state_obj": power_state,
                        "basis_state_dict": power_state.to_dict(),
                        "scaffold_family": "power",
                    },
                ],
            }
        family = basis_state.blocks[0].family
        return {
            "stats": {},
            "candidate_rows": [
                {
                    "scaffold_id": f"{family}:noop",
                    "expr": basis_state.compiled_expr,
                    "child_key": f"{family}:noop",
                    "proposal_key": f"{family}:noop",
                    "local_probe_mse": 1.0,
                    "local_fit_mse": 1.0,
                    "candidate_child_size": 2,
                    "basis_state_obj": basis_state,
                    "basis_state_dict": basis_state.to_dict(),
                    "scaffold_family": family,
                }
            ],
        }

    def _score_external_candidate_expr_fn(expr, **kwargs):
        candidate_state = kwargs["candidate_meta"]["basis_state_obj"]
        raw_mse = float(kwargs["candidate_meta"].get("local_probe_mse", 1.0))
        return {
            "expr": expr,
            "raw_mse": raw_mse,
            "fit_loss": raw_mse,
            "probe_loss": raw_mse,
            "basis_state_obj": candidate_state,
        }

    kwargs = _base_closure_search_kwargs()
    kwargs.update(
        {
            "closure_search_families": ["periodic", "power"],
            "closure_search_max_proposals": 16,
            "closure_search_exact_topk": 2,
            "closure_search_seed_exact_topk": 2,
            "closure_search_seed_beam_width": 2,
            "closure_search_beam_width": 2,
            "closure_search_pair_rescue_enable": False,
            "run_closure_search_pass_impl": _run_closure_search_pass_impl,
            "score_external_candidate_expr_fn": _score_external_candidate_expr_fn,
            "score_native_candidate_basis_state_fn": None,
        }
    )

    run_closure_search_pass(**kwargs)

    stats = kwargs["closure_search_stats"]
    assert len(seen_round_basis) == 2
    assert seen_round_basis[0] == ""
    assert seen_round_basis[1] == str(periodic_state.compiled_expr)
    assert int(stats["basis_state_round_updates"]) == 1
    assert int(stats["basis_state_round_commit_selected"]) == 1


def test_run_closure_search_pass_preserves_unexpanded_seed_state_after_first_update():
    periodic_state = basis_state_from_closure_candidate(
        family="periodic",
        scaffold_id="periodic:a",
        expr=("cos", ("var", 0)),
        anchor_node=None,
        scaffold_metadata={"form": "cos_base"},
        local_fit_mse=0.01,
        local_probe_mse=0.01,
        local_mapping_kind="direct_linear_head",
        direct_metadata={"feature_node": ("cos", ("var", 0))},
    )
    power_state = basis_state_from_closure_candidate(
        family="power",
        scaffold_id="power:b",
        expr=("sqr", ("var", 1)),
        anchor_node=None,
        scaffold_metadata={"form": "power_square"},
        local_fit_mse=0.02,
        local_probe_mse=0.02,
        local_mapping_kind="direct_power_head",
        direct_metadata={"power_inner_node": ("var", 1)},
    )
    improved_periodic = basis_state_from_closure_candidate(
        family="periodic",
        scaffold_id="periodic:a_improved",
        expr=("sin", ("var", 0)),
        anchor_node=None,
        scaffold_metadata={"form": "sin_base"},
        local_fit_mse=0.001,
        local_probe_mse=0.001,
        local_mapping_kind="direct_linear_head",
        direct_metadata={"feature_node": ("sin", ("var", 0))},
    )
    initial_beam = admit_basis_state_to_beam([], periodic_state, beam_width=2)
    initial_beam = admit_basis_state_to_beam(initial_beam, power_state, beam_width=2)
    seen_round_basis = []

    def _run_closure_search_pass_impl(**kwargs):
        proposal_context = kwargs.get("proposal_context")
        basis_state = proposal_context.basis_state if isinstance(proposal_context, ProposalContext) else None
        seen_round_basis.append(
            str(basis_state.compiled_expr) if isinstance(basis_state, type(periodic_state)) else ""
        )
        if not isinstance(basis_state, type(periodic_state)):
            return {"stats": {}, "candidate_rows": []}
        family = basis_state.blocks[0].family
        if family == "periodic":
            return {
                "stats": {},
                "candidate_rows": [
                    {
                        "scaffold_id": "periodic:improved",
                        "expr": improved_periodic.compiled_expr,
                        "child_key": "periodic:improved",
                        "proposal_key": "periodic:improved",
                        "local_probe_mse": 1.0e-3,
                        "local_fit_mse": 1.0e-3,
                        "candidate_child_size": 2,
                        "basis_state_obj": improved_periodic,
                        "feature_block_obj": improved_periodic.blocks[0],
                        "scaffold_family": "periodic",
                    }
                ],
            }
        return {
            "stats": {},
            "candidate_rows": [
                {
                    "scaffold_id": "power:noop",
                    "expr": basis_state.compiled_expr,
                    "child_key": "power:noop",
                    "proposal_key": "power:noop",
                    "local_probe_mse": 1.0,
                    "local_fit_mse": 1.0,
                    "candidate_child_size": 2,
                    "basis_state_obj": basis_state,
                    "feature_block_obj": basis_state.blocks[0],
                    "scaffold_family": "power",
                }
            ],
        }

    def _score_external_candidate_expr_fn(expr, **kwargs):
        candidate_state = kwargs["candidate_meta"]["basis_state_obj"]
        raw_mse = float(kwargs["candidate_meta"].get("local_probe_mse", 1.0))
        return {
            "expr": expr,
            "raw_mse": raw_mse,
            "fit_loss": raw_mse,
            "probe_loss": raw_mse,
            "basis_state_obj": candidate_state,
        }

    kwargs = _base_closure_search_kwargs()
    kwargs.update(
        {
            "closure_search_families": ["periodic", "power"],
            "closure_search_max_proposals": 16,
            "closure_search_exact_topk": 2,
            "closure_search_seed_exact_topk": 2,
            "closure_search_seed_beam_width": 2,
            "closure_search_beam_width": 2,
            "closure_search_pair_rescue_enable": False,
            "proposal_context": ProposalContext(
                basis_state=periodic_state,
                basis_state_beam=initial_beam,
                total_budget=16,
            ),
            "run_closure_search_pass_impl": _run_closure_search_pass_impl,
            "score_external_candidate_expr_fn": _score_external_candidate_expr_fn,
            "score_native_candidate_basis_state_fn": None,
        }
    )

    run_closure_search_pass(**kwargs)

    assert len(seen_round_basis) >= 2
    assert str(power_state.compiled_expr) in seen_round_basis[1:]


def test_run_closure_search_pass_pair_rescue_admits_complementary_pair_when_singletons_stall():
    base_state = basis_state_from_closure_candidate(
        family="affine",
        scaffold_id="affine:base",
        expr=("var", 2),
        anchor_node=None,
        scaffold_metadata={"form": "affine_base"},
        local_fit_mse=1.0,
        local_probe_mse=1.0,
        local_mapping_kind="direct_linear_head",
        direct_metadata={"feature_node": ("var", 2)},
    )
    state_a = basis_state_from_closure_candidate(
        family="periodic",
        scaffold_id="periodic:a",
        expr=("cos", ("var", 0)),
        anchor_node=None,
        scaffold_metadata={"form": "cos_base"},
        local_fit_mse=1.0,
        local_probe_mse=1.0,
        local_mapping_kind="direct_linear_head",
        direct_metadata={"feature_node": ("cos", ("var", 0)), "execution_mode": "exact_bound"},
    )
    state_b = basis_state_from_closure_candidate(
        family="power",
        scaffold_id="power:b",
        expr=("sqrt", ("var", 1)),
        anchor_node=None,
        scaffold_metadata={"form": "power_sqrt"},
        local_fit_mse=1.0,
        local_probe_mse=1.0,
        local_mapping_kind="direct_power_head",
        direct_metadata={"power_inner_node": ("var", 1), "execution_mode": "exact_bound"},
    )
    scored = []

    def _run_closure_search_pass_impl(**_kwargs):
        return {
            "stats": {},
            "candidate_rows": [
                {
                    "scaffold_id": "a",
                    "expr": ("cos", ("var", 0)),
                    "child_key": "a",
                    "proposal_key": "a",
                    "local_probe_mse": 0.1,
                    "local_fit_mse": 0.1,
                    "candidate_child_size": 2,
                    "basis_state_obj": state_a,
                    "feature_block_obj": state_a.blocks[0],
                    "scaffold_family": "periodic",
                    "execution_mode": "exact_bound",
                },
                {
                    "scaffold_id": "b",
                    "expr": ("sqrt", ("var", 1)),
                    "child_key": "b",
                    "proposal_key": "b",
                    "local_probe_mse": 0.2,
                    "local_fit_mse": 0.2,
                    "candidate_child_size": 2,
                    "basis_state_obj": state_b,
                    "feature_block_obj": state_b.blocks[0],
                    "scaffold_family": "power",
                    "execution_mode": "exact_bound",
                },
            ],
        }

    def _score_external_candidate_expr_fn(expr, **kwargs):
        proposal_key = str(kwargs["candidate_meta"].get("proposal_key", ""))
        scored.append(proposal_key)
        candidate_state = kwargs["candidate_meta"]["basis_state_obj"]
        raw_mse = 1.0e-6 if proposal_key.startswith("pair_rescue::") else 1.0
        return {
            "expr": expr,
            "raw_mse": raw_mse,
            "fit_loss": raw_mse,
            "probe_loss": raw_mse,
            "basis_state_obj": candidate_state,
        }

    kwargs = _base_closure_search_kwargs()
    kwargs.update(
        {
            "closure_search_exact_topk": 1,
            "closure_search_pair_rescue_enable": True,
            "closure_search_pair_rescue_topk": 2,
            "closure_search_pair_rescue_max_pairs": 1,
            "proposal_context": ProposalContext(
                basis_state=base_state,
                basis_state_beam=admit_basis_state_to_beam([], base_state, beam_width=4),
                total_budget=2,
            ),
            "run_closure_search_pass_impl": _run_closure_search_pass_impl,
            "score_external_candidate_expr_fn": _score_external_candidate_expr_fn,
            "score_native_candidate_basis_state_fn": None,
        }
    )

    run_closure_search_pass(**kwargs)

    stats = kwargs["closure_search_stats"]
    assert scored == ["a", "pair_rescue::a++b"]
    assert int(stats["basis_state_pair_rescue_rounds"]) == 1
    assert int(stats["basis_state_pair_rescue_scored"]) == 1
    assert int(stats["basis_state_pair_rescue_accepted"]) == 1
    accepted_pairs = list(stats.get("accepted_pair_events", []) or [])
    selected_pairs = list(stats.get("selected_pair_events", []) or [])
    assert accepted_pairs
    assert selected_pairs
    assert str(accepted_pairs[0].get("profile", "")) == "stall_expanded"
    assert str(selected_pairs[0].get("profile", "")) == "stall_expanded"
    assert "cross_family" in list(accepted_pairs[0].get("relation_tags", []) or [])
    assert list(accepted_pairs[0].get("pair_member_sources", []) or []) == [
        "exact_scored_singleton",
        "prioritized_only",
    ]
    beam_rows = list(stats["basis_state_beam"] or [])
    assert any(int(row.get("block_count", 0) or 0) >= 3 for row in beam_rows)


def test_run_closure_search_pass_pair_rescue_can_override_weak_early_singleton():
    base_state = basis_state_from_closure_candidate(
        family="affine",
        scaffold_id="affine:base",
        expr=("var", 2),
        anchor_node=None,
        scaffold_metadata={"form": "affine_base"},
        local_fit_mse=1.015,
        local_probe_mse=1.015,
        local_mapping_kind="direct_linear_head",
        direct_metadata={"feature_node": ("var", 2)},
    )
    state_a = basis_state_from_closure_candidate(
        family="periodic",
        scaffold_id="periodic:a",
        expr=("cos", ("var", 0)),
        anchor_node=None,
        scaffold_metadata={"form": "cos_base"},
        local_fit_mse=1.0,
        local_probe_mse=1.0,
        local_mapping_kind="direct_linear_head",
        direct_metadata={"feature_node": ("cos", ("var", 0)), "execution_mode": "exact_bound"},
    )
    state_b = basis_state_from_closure_candidate(
        family="power",
        scaffold_id="power:b",
        expr=("sqrt", ("var", 1)),
        anchor_node=None,
        scaffold_metadata={"form": "power_sqrt"},
        local_fit_mse=1.0,
        local_probe_mse=1.0,
        local_mapping_kind="direct_power_head",
        direct_metadata={"power_inner_node": ("var", 1), "execution_mode": "exact_bound"},
    )
    scored = []

    def _run_closure_search_pass_impl(**_kwargs):
        return {
            "stats": {},
            "candidate_rows": [
                {
                    "scaffold_id": "a",
                    "expr": ("cos", ("var", 0)),
                    "child_key": "a",
                    "proposal_key": "a",
                    "local_probe_mse": 1.0,
                    "local_fit_mse": 1.0,
                    "candidate_child_size": 2,
                    "basis_state_obj": state_a,
                    "feature_block_obj": state_a.blocks[0],
                    "scaffold_family": "periodic",
                    "execution_mode": "exact_bound",
                },
                {
                    "scaffold_id": "b",
                    "expr": ("sqrt", ("var", 1)),
                    "child_key": "b",
                    "proposal_key": "b",
                    "local_probe_mse": 1.1,
                    "local_fit_mse": 1.1,
                    "candidate_child_size": 2,
                    "basis_state_obj": state_b,
                    "feature_block_obj": state_b.blocks[0],
                    "scaffold_family": "power",
                    "execution_mode": "exact_bound",
                },
            ],
        }

    def _score_external_candidate_expr_fn(expr, **kwargs):
        proposal_key = str(kwargs["candidate_meta"].get("proposal_key", ""))
        scored.append(proposal_key)
        candidate_state = kwargs["candidate_meta"]["basis_state_obj"]
        if proposal_key == "a":
            raw_mse = 1.0
        elif proposal_key == "pair_rescue::a++b":
            raw_mse = 0.8
        else:
            raw_mse = 1.1
        return {
            "expr": expr,
            "raw_mse": raw_mse,
            "fit_loss": raw_mse,
            "probe_loss": raw_mse,
            "basis_state_obj": candidate_state,
        }

    kwargs = _base_closure_search_kwargs()
    kwargs.update(
        {
            "closure_search_exact_topk": 1,
            "closure_search_debug_topk": 2,
            "closure_search_pair_rescue_enable": True,
            "closure_search_pair_rescue_topk": 2,
            "closure_search_pair_rescue_max_pairs": 1,
            "proposal_context": ProposalContext(
                basis_state=base_state,
                basis_state_beam=admit_basis_state_to_beam([], base_state, beam_width=4),
                total_budget=2,
            ),
            "run_closure_search_pass_impl": _run_closure_search_pass_impl,
            "score_external_candidate_expr_fn": _score_external_candidate_expr_fn,
            "score_native_candidate_basis_state_fn": None,
        }
    )

    run_closure_search_pass(**kwargs)

    stats = kwargs["closure_search_stats"]
    assert scored == ["a", "pair_rescue::a++b"]
    assert int(stats["basis_state_pair_rescue_rounds"]) == 1
    assert int(stats["basis_state_pair_rescue_scored"]) == 1
    assert int(stats["basis_state_pair_rescue_accepted"]) == 1
    assert int(stats["basis_state_round_commit_selected_pair"]) == 1
    accepted_pairs = list(stats.get("accepted_pair_events", []) or [])
    selected_pairs = list(stats.get("selected_pair_events", []) or [])
    round_summaries = list(stats.get("debug_round_summaries", []) or [])
    assert accepted_pairs
    assert selected_pairs
    assert str(selected_pairs[0].get("profile", "")) == "stall_expanded"
    assert round_summaries
    assert str(round_summaries[0].get("early_selected_commit_kind", "")) == "singleton"
    assert str(round_summaries[0].get("selected_commit_kind", "")) == "pair"
    assert "weak_singleton_gain" in list(round_summaries[0].get("pair_rescue_trigger_reasons", []) or [])


def test_run_closure_search_pass_pair_rescue_rechooses_over_early_and_rescue_commits():
    base_state = basis_state_from_closure_candidate(
        family="affine",
        scaffold_id="affine:base",
        expr=("var", 2),
        anchor_node=None,
        scaffold_metadata={"form": "affine_base"},
        local_fit_mse=1.015,
        local_probe_mse=1.015,
        local_mapping_kind="direct_linear_head",
        direct_metadata={"feature_node": ("var", 2)},
    )
    state_a = basis_state_from_closure_candidate(
        family="periodic",
        scaffold_id="periodic:a",
        expr=("cos", ("var", 0)),
        anchor_node=None,
        scaffold_metadata={"form": "cos_base"},
        local_fit_mse=1.0,
        local_probe_mse=1.0,
        local_mapping_kind="direct_linear_head",
        direct_metadata={"feature_node": ("cos", ("var", 0)), "execution_mode": "exact_bound"},
    )
    state_b = basis_state_from_closure_candidate(
        family="power",
        scaffold_id="power:b",
        expr=("sqrt", ("var", 1)),
        anchor_node=None,
        scaffold_metadata={"form": "power_sqrt"},
        local_fit_mse=1.0,
        local_probe_mse=1.0,
        local_mapping_kind="direct_power_head",
        direct_metadata={"power_inner_node": ("var", 1), "execution_mode": "exact_bound"},
    )
    scored = []

    def _run_closure_search_pass_impl(**_kwargs):
        return {
            "stats": {},
            "candidate_rows": [
                {
                    "scaffold_id": "a",
                    "expr": ("cos", ("var", 0)),
                    "child_key": "a",
                    "proposal_key": "a",
                    "local_probe_mse": 1.0,
                    "local_fit_mse": 1.0,
                    "candidate_child_size": 2,
                    "basis_state_obj": state_a,
                    "feature_block_obj": state_a.blocks[0],
                    "scaffold_family": "periodic",
                    "execution_mode": "exact_bound",
                },
                {
                    "scaffold_id": "b",
                    "expr": ("sqrt", ("var", 1)),
                    "child_key": "b",
                    "proposal_key": "b",
                    "local_probe_mse": 1.1,
                    "local_fit_mse": 1.1,
                    "candidate_child_size": 2,
                    "basis_state_obj": state_b,
                    "feature_block_obj": state_b.blocks[0],
                    "scaffold_family": "power",
                    "execution_mode": "exact_bound",
                },
            ],
        }

    def _score_external_candidate_expr_fn(expr, **kwargs):
        proposal_key = str(kwargs["candidate_meta"].get("proposal_key", ""))
        scored.append(proposal_key)
        candidate_state = kwargs["candidate_meta"]["basis_state_obj"]
        if proposal_key == "a":
            raw_mse = 1.0
        elif proposal_key == "pair_rescue::a++b":
            raw_mse = 1.01
        else:
            raw_mse = 1.1
        return {
            "expr": expr,
            "raw_mse": raw_mse,
            "fit_loss": raw_mse,
            "probe_loss": raw_mse,
            "basis_state_obj": candidate_state,
        }

    kwargs = _base_closure_search_kwargs()
    kwargs.update(
        {
            "closure_search_exact_topk": 1,
            "closure_search_debug_topk": 2,
            "closure_search_pair_rescue_enable": True,
            "closure_search_pair_rescue_topk": 2,
            "closure_search_pair_rescue_max_pairs": 1,
            "proposal_context": ProposalContext(
                basis_state=base_state,
                basis_state_beam=admit_basis_state_to_beam([], base_state, beam_width=4),
                total_budget=2,
            ),
            "run_closure_search_pass_impl": _run_closure_search_pass_impl,
            "score_external_candidate_expr_fn": _score_external_candidate_expr_fn,
            "score_native_candidate_basis_state_fn": None,
        }
    )

    run_closure_search_pass(**kwargs)

    stats = kwargs["closure_search_stats"]
    assert scored == ["a", "pair_rescue::a++b"]
    assert int(stats["basis_state_pair_rescue_rounds"]) == 1
    assert int(stats["basis_state_pair_rescue_scored"]) == 1
    assert int(stats["basis_state_pair_rescue_accepted"]) == 1
    assert int(stats["basis_state_round_commit_selected_singleton"]) == 1
    accepted_pairs = list(stats.get("accepted_pair_events", []) or [])
    selected_pairs = list(stats.get("selected_pair_events", []) or [])
    round_summaries = list(stats.get("debug_round_summaries", []) or [])
    assert len(accepted_pairs) == 1
    assert str(accepted_pairs[0].get("profile", "")) == "stall_expanded"
    assert selected_pairs == []
    assert round_summaries
    assert str(round_summaries[0].get("early_selected_commit_kind", "")) == "singleton"
    assert str(round_summaries[0].get("selected_commit_kind", "")) == "singleton"
    assert "weak_singleton_gain" in list(round_summaries[0].get("pair_rescue_trigger_reasons", []) or [])


def test_run_closure_search_pass_pair_rescue_skips_pair_already_attempted_by_normal():
    base_state = basis_state_from_closure_candidate(
        family="affine",
        scaffold_id="affine:base",
        expr=("var", 2),
        anchor_node=None,
        scaffold_metadata={"form": "affine_base"},
        local_fit_mse=1.015,
        local_probe_mse=1.015,
        local_mapping_kind="direct_linear_head",
        direct_metadata={"feature_node": ("var", 2)},
    )
    state_a = basis_state_from_closure_candidate(
        family="periodic",
        scaffold_id="periodic:a",
        expr=("cos", ("var", 0)),
        anchor_node=None,
        scaffold_metadata={"form": "cos_base"},
        local_fit_mse=1.0,
        local_probe_mse=1.0,
        local_mapping_kind="direct_linear_head",
        direct_metadata={"feature_node": ("cos", ("var", 0)), "execution_mode": "exact_bound"},
    )
    state_b = basis_state_from_closure_candidate(
        family="power",
        scaffold_id="power:b",
        expr=("sqrt", ("var", 1)),
        anchor_node=None,
        scaffold_metadata={"form": "power_sqrt"},
        local_fit_mse=1.2,
        local_probe_mse=1.2,
        local_mapping_kind="direct_power_head",
        direct_metadata={"power_inner_node": ("var", 1), "execution_mode": "exact_bound"},
    )
    scored = []

    def _run_closure_search_pass_impl(**_kwargs):
        return {
            "stats": {},
            "candidate_rows": [
                {
                    "scaffold_id": "a",
                    "expr": ("cos", ("var", 0)),
                    "child_key": "a",
                    "proposal_key": "a",
                    "local_probe_mse": 1.0,
                    "local_fit_mse": 1.0,
                    "candidate_child_size": 2,
                    "basis_state_obj": state_a,
                    "feature_block_obj": state_a.blocks[0],
                    "scaffold_family": "periodic",
                    "execution_mode": "exact_bound",
                },
                {
                    "scaffold_id": "b",
                    "expr": ("sqrt", ("var", 1)),
                    "child_key": "b",
                    "proposal_key": "b",
                    "local_probe_mse": 1.2,
                    "local_fit_mse": 1.2,
                    "candidate_child_size": 2,
                    "basis_state_obj": state_b,
                    "feature_block_obj": state_b.blocks[0],
                    "scaffold_family": "power",
                    "execution_mode": "exact_bound",
                },
            ],
        }

    def _score_external_candidate_expr_fn(expr, **kwargs):
        proposal_key = str(kwargs["candidate_meta"].get("proposal_key", ""))
        scored.append(proposal_key)
        candidate_state = kwargs["candidate_meta"]["basis_state_obj"]
        if proposal_key == "a":
            raw_mse = 1.0
        elif proposal_key == "b":
            raw_mse = 1.2
        elif proposal_key == "pair_normal::a++b":
            raw_mse = 1.01
        else:
            raw_mse = 0.5
        return {
            "expr": expr,
            "raw_mse": raw_mse,
            "fit_loss": raw_mse,
            "probe_loss": raw_mse,
            "basis_state_obj": candidate_state,
        }

    kwargs = _base_closure_search_kwargs()
    kwargs.update(
        {
            "closure_search_exact_topk": 2,
            "closure_search_debug_topk": 2,
            "closure_search_pair_normal_enable": True,
            "closure_search_pair_normal_topk": 2,
            "closure_search_pair_normal_max_pairs": 1,
            "closure_search_pair_rescue_enable": True,
            "closure_search_pair_rescue_topk": 2,
            "closure_search_pair_rescue_max_pairs": 1,
            "proposal_context": ProposalContext(
                basis_state=base_state,
                basis_state_beam=admit_basis_state_to_beam([], base_state, beam_width=4),
                total_budget=2,
            ),
            "run_closure_search_pass_impl": _run_closure_search_pass_impl,
            "score_external_candidate_expr_fn": _score_external_candidate_expr_fn,
            "score_native_candidate_basis_state_fn": None,
        }
    )

    run_closure_search_pass(**kwargs)

    stats = kwargs["closure_search_stats"]
    assert scored[:3] == ["a", "b", "pair_normal::a++b"]
    assert int(stats["basis_state_pair_normal_scored"]) == 1
    assert int(stats["basis_state_pair_normal_accepted"]) == 1
    accepted_pairs = list(stats.get("accepted_pair_events", []) or [])
    selected_pairs = list(stats.get("selected_pair_events", []) or [])
    pair_attempts = list(stats.get("debug_pair_attempts", []) or [])
    round_summaries = list(stats.get("debug_round_summaries", []) or [])
    round1_pairs = [row for row in accepted_pairs if int(row.get("round", 0) or 0) == 1]
    round1_selected_pairs = [row for row in selected_pairs if int(row.get("round", 0) or 0) == 1]
    assert len(round1_pairs) == 1
    assert str(round1_pairs[0].get("profile", "")) == "normal"
    assert round1_selected_pairs == []
    assert not any(
        int(row.get("round", 0) or 0) == 1 and str(row.get("stage", "")) == "pair_rescue"
        for row in pair_attempts
    )
    assert round_summaries
    assert "weak_singleton_gain" in list(round_summaries[0].get("pair_rescue_trigger_reasons", []) or [])
    assert "close_accepted_pair" in list(round_summaries[0].get("pair_rescue_trigger_reasons", []) or [])


def test_run_closure_search_pass_pair_rescue_can_materialize_prioritized_member_without_basis_state():
    base_state = basis_state_from_closure_candidate(
        family="affine",
        scaffold_id="affine:base",
        expr=("var", 2),
        anchor_node=None,
        scaffold_metadata={"form": "affine_base"},
        local_fit_mse=1.015,
        local_probe_mse=1.015,
        local_mapping_kind="direct_linear_head",
        direct_metadata={"feature_node": ("var", 2)},
    )
    state_b = basis_state_from_closure_candidate(
        family="power",
        scaffold_id="power:b",
        expr=("sqrt", ("var", 1)),
        anchor_node=None,
        scaffold_metadata={"form": "power_sqrt"},
        local_fit_mse=1.0,
        local_probe_mse=1.0,
        local_mapping_kind="direct_power_head",
        direct_metadata={"power_inner_node": ("var", 1), "execution_mode": "exact_bound"},
    )
    state_c = basis_state_from_closure_candidate(
        family="periodic",
        scaffold_id="periodic:c",
        expr=("cos", ("var", 0)),
        anchor_node=None,
        scaffold_metadata={"form": "cos_base"},
        local_fit_mse=1.1,
        local_probe_mse=1.1,
        local_mapping_kind="direct_linear_head",
        direct_metadata={"feature_node": ("cos", ("var", 0)), "execution_mode": "exact_bound"},
    )
    scored = []

    def _run_closure_search_pass_impl(**_kwargs):
        return {
            "stats": {},
            "candidate_rows": [
                {
                    "scaffold_id": "b",
                    "expr": ("sqrt", ("var", 1)),
                    "child_key": "b",
                    "proposal_key": "b",
                    "local_probe_mse": 1.0,
                    "local_fit_mse": 1.0,
                    "candidate_child_size": 2,
                    "basis_state_obj": state_b,
                    "feature_block_obj": state_b.blocks[0],
                    "scaffold_family": "power",
                    "execution_mode": "exact_bound",
                },
                {
                    "scaffold_id": "c",
                    "expr": ("cos", ("var", 0)),
                    "child_key": "c",
                    "proposal_key": "c",
                    "local_probe_mse": 1.0e-10,
                    "local_fit_mse": 1.0e-10,
                    "candidate_child_size": 2,
                    "feature_block_obj": state_c.blocks[0],
                    "scaffold_family": "periodic",
                    "execution_mode": "exact_bound",
                },
            ],
        }

    def _score_external_candidate_expr_fn(expr, **kwargs):
        proposal_key = str(kwargs["candidate_meta"].get("proposal_key", ""))
        scored.append(proposal_key)
        candidate_state = kwargs["candidate_meta"]["basis_state_obj"]
        if proposal_key == "c":
            raw_mse = 1.1
        elif proposal_key == "b":
            raw_mse = 1.0
        elif proposal_key in {"pair_rescue::b++c", "pair_rescue::c++b"}:
            raw_mse = 0.5
        else:
            raw_mse = 1.1
        return {
            "expr": expr,
            "raw_mse": raw_mse,
            "fit_loss": raw_mse,
            "probe_loss": raw_mse,
            "basis_state_obj": candidate_state,
        }

    kwargs = _base_closure_search_kwargs()
    kwargs.update(
        {
            "closure_search_exact_topk": 1,
            "closure_search_debug_topk": 2,
            "closure_search_pair_rescue_enable": True,
            "closure_search_pair_rescue_topk": 2,
            "closure_search_pair_rescue_max_pairs": 1,
            "proposal_context": ProposalContext(
                basis_state=base_state,
                basis_state_beam=admit_basis_state_to_beam([], base_state, beam_width=4),
                total_budget=2,
            ),
            "run_closure_search_pass_impl": _run_closure_search_pass_impl,
            "score_external_candidate_expr_fn": _score_external_candidate_expr_fn,
            "score_native_candidate_basis_state_fn": None,
        }
    )

    run_closure_search_pass(**kwargs)

    stats = kwargs["closure_search_stats"]
    assert scored[:2] == ["c", "b"]
    assert any(str(key).startswith("pair_rescue::") for key in scored)
    assert int(stats["basis_state_pair_rescue_rounds"]) == 1
    assert int(stats["basis_state_pair_rescue_scored"]) == 1
    assert int(stats["basis_state_pair_rescue_accepted"]) == 1
    accepted_pairs = list(stats.get("accepted_pair_events", []) or [])
    selected_pairs = list(stats.get("selected_pair_events", []) or [])
    assert accepted_pairs
    assert selected_pairs
    assert str(accepted_pairs[0].get("profile", "")) == "stall_expanded"
    assert sorted(list(accepted_pairs[0].get("pair_member_sources", []) or [])) == [
        "exact_scored_singleton",
        "prioritized_only",
    ]


def test_run_closure_search_pass_pair_rescue_keeps_companion_slots_beyond_exact_anchors():
    base_state = basis_state_from_closure_candidate(
        family="affine",
        scaffold_id="affine:base",
        expr=("var", 2),
        anchor_node=None,
        scaffold_metadata={"form": "affine_base"},
        local_fit_mse=1.015,
        local_probe_mse=1.015,
        local_mapping_kind="direct_linear_head",
        direct_metadata={"feature_node": ("var", 2)},
    )
    state_p1 = basis_state_from_closure_candidate(
        family="periodic",
        scaffold_id="periodic:p1",
        expr=("cos", ("var", 0)),
        anchor_node=None,
        scaffold_metadata={"form": "cos_base"},
        local_fit_mse=1.0,
        local_probe_mse=1.0,
        local_mapping_kind="direct_linear_head",
        direct_metadata={"feature_node": ("cos", ("var", 0)), "execution_mode": "exact_bound"},
    )
    state_p2 = basis_state_from_closure_candidate(
        family="periodic",
        scaffold_id="periodic:p2",
        expr=("sin", ("var", 0)),
        anchor_node=None,
        scaffold_metadata={"form": "sin_base"},
        local_fit_mse=1.01,
        local_probe_mse=1.01,
        local_mapping_kind="direct_linear_head",
        direct_metadata={"feature_node": ("sin", ("var", 0)), "execution_mode": "exact_bound"},
    )
    state_q1 = basis_state_from_closure_candidate(
        family="power",
        scaffold_id="power:q1",
        expr=("sqrt", ("var", 1)),
        anchor_node=None,
        scaffold_metadata={"form": "power_sqrt"},
        local_fit_mse=1.1,
        local_probe_mse=1.1,
        local_mapping_kind="direct_power_head",
        direct_metadata={"power_inner_node": ("var", 1), "execution_mode": "exact_bound"},
    )
    scored = []

    def _run_closure_search_pass_impl(**_kwargs):
        return {
            "stats": {},
            "candidate_rows": [
                {
                    "scaffold_id": "p1",
                    "expr": ("cos", ("var", 0)),
                    "child_key": "p1",
                    "proposal_key": "p1",
                    "local_probe_mse": 1.0,
                    "local_fit_mse": 1.0,
                    "candidate_child_size": 2,
                    "basis_state_obj": state_p1,
                    "feature_block_obj": state_p1.blocks[0],
                    "scaffold_family": "periodic",
                    "execution_mode": "exact_bound",
                },
                {
                    "scaffold_id": "p2",
                    "expr": ("sin", ("var", 0)),
                    "child_key": "p2",
                    "proposal_key": "p2",
                    "local_probe_mse": 1.01,
                    "local_fit_mse": 1.01,
                    "candidate_child_size": 2,
                    "basis_state_obj": state_p2,
                    "feature_block_obj": state_p2.blocks[0],
                    "scaffold_family": "periodic",
                    "execution_mode": "exact_bound",
                },
                {
                    "scaffold_id": "q1",
                    "expr": ("sqrt", ("var", 1)),
                    "child_key": "q1",
                    "proposal_key": "q1",
                    "local_probe_mse": 1.02,
                    "local_fit_mse": 1.02,
                    "candidate_child_size": 2,
                    "feature_block_obj": state_q1.blocks[0],
                    "scaffold_family": "power",
                    "execution_mode": "exact_bound",
                },
            ],
        }

    def _score_external_candidate_expr_fn(expr, **kwargs):
        proposal_key = str(kwargs["candidate_meta"].get("proposal_key", ""))
        scored.append(proposal_key)
        candidate_state = kwargs["candidate_meta"]["basis_state_obj"]
        if proposal_key == "p1":
            raw_mse = 1.0
        elif proposal_key == "p2":
            raw_mse = 1.01
        elif proposal_key == "q1":
            raw_mse = 1.02
        elif proposal_key == "pair_normal::p1++p2":
            raw_mse = 1.005
        elif proposal_key == "pair_rescue::p1++q1":
            raw_mse = 0.8
        else:
            raw_mse = 1.2
        return {
            "expr": expr,
            "raw_mse": raw_mse,
            "fit_loss": raw_mse,
            "probe_loss": raw_mse,
            "basis_state_obj": candidate_state,
        }

    kwargs = _base_closure_search_kwargs()
    kwargs.update(
        {
            "closure_search_exact_topk": 2,
            "closure_search_debug_topk": 4,
            "closure_search_pair_normal_enable": True,
            "closure_search_pair_normal_topk": 2,
            "closure_search_pair_normal_max_pairs": 1,
            "closure_search_pair_rescue_enable": True,
            "closure_search_pair_rescue_topk": 2,
            "closure_search_pair_rescue_max_pairs": 1,
            "proposal_context": ProposalContext(
                basis_state=base_state,
                basis_state_beam=admit_basis_state_to_beam([], base_state, beam_width=4),
                total_budget=3,
            ),
            "run_closure_search_pass_impl": _run_closure_search_pass_impl,
            "score_external_candidate_expr_fn": _score_external_candidate_expr_fn,
            "score_native_candidate_basis_state_fn": None,
        }
    )

    run_closure_search_pass(**kwargs)

    stats = kwargs["closure_search_stats"]
    assert scored[:2] == ["p1", "p2"]
    assert any(str(key).startswith("pair_normal::") for key in scored)
    assert any(str(key).startswith("pair_rescue::") for key in scored)
    assert int(stats["basis_state_pair_normal_scored"]) == 1
    assert int(stats["basis_state_pair_rescue_scored"]) >= 1
    assert int(stats["basis_state_pair_rescue_accepted"]) == 1
    accepted_pairs = list(stats.get("accepted_pair_events", []) or [])
    rescue_pairs = [row for row in accepted_pairs if str(row.get("profile", "")) == "stall_expanded"]
    assert len(rescue_pairs) == 1
    assert sorted(list(rescue_pairs[0].get("pair_member_sources", []) or [])) == [
        "exact_scored_singleton",
        "prioritized_only",
    ]


def test_run_closure_search_pass_pair_normal_admits_exact_scored_pair_when_enabled():
    base_state = basis_state_from_closure_candidate(
        family="periodic",
        scaffold_id="base",
        expr=("var", 0),
        anchor_node=None,
        scaffold_metadata={"form": "base"},
        local_fit_mse=0.05,
        local_probe_mse=0.05,
        local_mapping_kind="direct_linear_head",
    )
    state_a = basis_state_from_closure_candidate(
        family="periodic",
        scaffold_id="a",
        expr=("cos", ("var", 0)),
        anchor_node=None,
        scaffold_metadata={"form": "cos_base"},
        local_fit_mse=0.1,
        local_probe_mse=0.1,
        local_mapping_kind="direct_linear_head",
        direct_metadata={"feature_node": ("cos", ("var", 0))},
    )
    state_b = basis_state_from_closure_candidate(
        family="power",
        scaffold_id="b",
        expr=("mul", ("var", 1), ("cos", ("var", 0))),
        anchor_node=("var", 1),
        scaffold_metadata={"form": "power_companion"},
        local_fit_mse=0.2,
        local_probe_mse=0.2,
        local_mapping_kind="direct_power_head",
        direct_metadata={"feature_node": ("cos", ("var", 0)), "anchor_node": ("var", 1)},
    )
    scored = []

    def _run_closure_search_pass_impl(**_kwargs):
        return {
            "stats": {},
            "candidate_rows": [
                {
                    "scaffold_id": "a",
                    "expr": ("cos", ("var", 0)),
                    "child_key": "a",
                    "proposal_key": "a",
                    "local_probe_mse": 0.1,
                    "local_fit_mse": 0.1,
                    "candidate_child_size": 2,
                    "basis_state_obj": state_a,
                    "feature_block_obj": state_a.blocks[0],
                    "scaffold_family": "periodic",
                    "execution_mode": "exact_bound",
                },
                {
                    "scaffold_id": "b",
                    "expr": ("mul", ("var", 1), ("cos", ("var", 0))),
                    "child_key": "b",
                    "proposal_key": "b",
                    "local_probe_mse": 0.2,
                    "local_fit_mse": 0.2,
                    "candidate_child_size": 2,
                    "basis_state_obj": state_b,
                    "feature_block_obj": state_b.blocks[0],
                    "scaffold_family": "power",
                    "execution_mode": "exact_bound",
                },
            ],
        }

    def _score_external_candidate_expr_fn(expr, **kwargs):
        proposal_key = str(kwargs["candidate_meta"].get("proposal_key", ""))
        scored.append(proposal_key)
        candidate_state = kwargs["candidate_meta"]["basis_state_obj"]
        raw_mse = 1.0e-6 if proposal_key.startswith("pair_normal::") else 1.0
        return {
            "expr": expr,
            "raw_mse": raw_mse,
            "fit_loss": raw_mse,
            "probe_loss": raw_mse,
            "basis_state_obj": candidate_state,
        }

    kwargs = _base_closure_search_kwargs()
    kwargs.update(
        {
            "closure_search_exact_topk": 2,
            "closure_search_pair_normal_enable": True,
            "closure_search_pair_normal_topk": 2,
            "closure_search_pair_normal_max_pairs": 1,
            "closure_search_pair_rescue_enable": False,
            "proposal_context": ProposalContext(
                basis_state=base_state,
                basis_state_beam=admit_basis_state_to_beam([], base_state, beam_width=4),
                total_budget=2,
            ),
            "run_closure_search_pass_impl": _run_closure_search_pass_impl,
            "score_external_candidate_expr_fn": _score_external_candidate_expr_fn,
            "score_native_candidate_basis_state_fn": None,
        }
    )

    run_closure_search_pass(**kwargs)

    stats = kwargs["closure_search_stats"]
    assert scored == ["a", "b", "pair_normal::a++b"]
    assert int(stats["basis_state_pair_normal_rounds"]) == 1
    assert int(stats["basis_state_pair_normal_scored"]) == 1
    assert int(stats["basis_state_pair_normal_accepted"]) == 1
    accepted_pairs = list(stats.get("accepted_pair_events", []) or [])
    selected_pairs = list(stats.get("selected_pair_events", []) or [])
    assert accepted_pairs
    assert selected_pairs
    assert str(accepted_pairs[0].get("profile", "")) == "normal"
    assert str(selected_pairs[0].get("profile", "")) == "normal"
    assert "cross_family" in list(accepted_pairs[0].get("relation_tags", []) or [])
    assert list(accepted_pairs[0].get("pair_member_sources", []) or []) == [
        "exact_scored_singleton",
        "exact_scored_singleton",
    ]


def test_run_closure_search_pass_logs_accepted_pair_even_when_singleton_wins_round():
    base_state = basis_state_from_closure_candidate(
        family="periodic",
        scaffold_id="base",
        expr=("var", 0),
        anchor_node=None,
        scaffold_metadata={"form": "base"},
        local_fit_mse=2.0,
        local_probe_mse=2.0,
        local_mapping_kind="direct_linear_head",
    )
    state_a = basis_state_from_closure_candidate(
        family="periodic",
        scaffold_id="a",
        expr=("cos", ("var", 0)),
        anchor_node=None,
        scaffold_metadata={"form": "cos_base"},
        local_fit_mse=1.0,
        local_probe_mse=1.0,
        local_mapping_kind="direct_linear_head",
        direct_metadata={"feature_node": ("cos", ("var", 0))},
    )
    state_b = basis_state_from_closure_candidate(
        family="power",
        scaffold_id="b",
        expr=("mul", ("var", 1), ("cos", ("var", 0))),
        anchor_node=("var", 1),
        scaffold_metadata={"form": "power_companion"},
        local_fit_mse=1.2,
        local_probe_mse=1.2,
        local_mapping_kind="direct_power_head",
        direct_metadata={"feature_node": ("cos", ("var", 0)), "anchor_node": ("var", 1)},
    )

    def _run_closure_search_pass_impl(**_kwargs):
        return {
            "stats": {},
            "candidate_rows": [
                {
                    "scaffold_id": "a",
                    "expr": ("cos", ("var", 0)),
                    "child_key": "a",
                    "proposal_key": "a",
                    "local_probe_mse": 1.0,
                    "local_fit_mse": 1.0,
                    "candidate_child_size": 2,
                    "basis_state_obj": state_a,
                    "feature_block_obj": state_a.blocks[0],
                    "scaffold_family": "periodic",
                    "execution_mode": "exact_bound",
                },
                {
                    "scaffold_id": "b",
                    "expr": ("mul", ("var", 1), ("cos", ("var", 0))),
                    "child_key": "b",
                    "proposal_key": "b",
                    "local_probe_mse": 1.2,
                    "local_fit_mse": 1.2,
                    "candidate_child_size": 2,
                    "basis_state_obj": state_b,
                    "feature_block_obj": state_b.blocks[0],
                    "scaffold_family": "power",
                    "execution_mode": "exact_bound",
                },
            ],
        }

    def _score_external_candidate_expr_fn(expr, **kwargs):
        proposal_key = str(kwargs["candidate_meta"].get("proposal_key", ""))
        candidate_state = kwargs["candidate_meta"]["basis_state_obj"]
        if proposal_key == "a":
            raw_mse = 1.0
        elif proposal_key == "b":
            raw_mse = 1.2
        else:
            raw_mse = 1.01
        return {
            "expr": expr,
            "raw_mse": raw_mse,
            "fit_loss": raw_mse,
            "probe_loss": raw_mse,
            "basis_state_obj": candidate_state,
        }

    kwargs = _base_closure_search_kwargs()
    kwargs.update(
        {
            "closure_search_exact_topk": 2,
            "closure_search_pair_normal_enable": True,
            "closure_search_pair_normal_topk": 2,
            "closure_search_pair_normal_max_pairs": 1,
            "closure_search_pair_rescue_enable": False,
            "proposal_context": ProposalContext(
                basis_state=base_state,
                basis_state_beam=admit_basis_state_to_beam([], base_state, beam_width=4),
                total_budget=2,
            ),
            "run_closure_search_pass_impl": _run_closure_search_pass_impl,
            "score_external_candidate_expr_fn": _score_external_candidate_expr_fn,
            "score_native_candidate_basis_state_fn": None,
        }
    )

    run_closure_search_pass(**kwargs)

    stats = kwargs["closure_search_stats"]
    assert int(stats["basis_state_pair_normal_accepted"]) == 1
    assert int(stats["basis_state_round_commit_selected_singleton"]) == 1
    accepted_pairs = list(stats.get("accepted_pair_events", []) or [])
    selected_pairs = list(stats.get("selected_pair_events", []) or [])
    assert len(accepted_pairs) == 1
    assert str(accepted_pairs[0].get("profile", "")) == "normal"
    assert selected_pairs == []


def test_run_closure_search_pass_debug_trace_exports_preview_and_exact_rows():
    preview_state = basis_state_from_closure_candidate(
        family="periodic",
        scaffold_id="periodic:cos",
        expr=("cos", ("var", 0)),
        anchor_node=None,
        scaffold_metadata={"form": "cos_base"},
        local_fit_mse=0.1,
        local_probe_mse=0.1,
        local_mapping_kind="direct_linear_head",
        direct_metadata={"feature_node": ("cos", ("var", 0)), "execution_mode": "exact_bound"},
    )

    def _run_closure_search_pass_impl(**_kwargs):
        return {
            "stats": {},
            "candidate_rows": [
                {
                    "scaffold_id": "p1",
                    "expr": ("cos", ("var", 0)),
                    "child_key": "p1",
                    "proposal_key": "p1",
                    "local_probe_mse": 0.10,
                    "local_fit_mse": 0.10,
                    "candidate_child_size": 2,
                    "basis_state_obj": preview_state,
                    "feature_block_obj": preview_state.blocks[0],
                    "scaffold_family": "periodic",
                    "operator_id": "periodic:cos",
                    "execution_mode": "exact_bound",
                },
            ],
        }

    def _score_external_candidate_expr_fn(expr, **kwargs):
        candidate_state = kwargs["candidate_meta"]["basis_state_obj"]
        return {
            "expr": expr,
            "raw_mse": 1.0e-6,
            "fit_loss": 1.0e-6,
            "probe_loss": 1.0e-6,
            "basis_state_obj": candidate_state,
        }

    kwargs = _base_closure_search_kwargs()
    kwargs.update(
        {
            "closure_search_debug_topk": 2,
            "run_closure_search_pass_impl": _run_closure_search_pass_impl,
            "score_external_candidate_expr_fn": _score_external_candidate_expr_fn,
            "score_native_candidate_basis_state_fn": None,
        }
    )

    run_closure_search_pass(**kwargs)

    stats = kwargs["closure_search_stats"]
    preview_rows = list(stats.get("debug_preview_rows", []) or [])
    exact_rows = list(stats.get("debug_exact_rows", []) or [])
    round_summaries = list(stats.get("debug_round_summaries", []) or [])
    assert preview_rows
    assert exact_rows
    assert round_summaries
    assert str(preview_rows[0].get("proposal_key", "")) == "p1"
    assert str(exact_rows[0].get("proposal_key", "")) == "p1"
    assert bool(exact_rows[0].get("accepted", False)) is True
    assert int(round_summaries[0]["raw_family_counts"]["periodic"]) == 1
    assert int(round_summaries[0]["expr_family_counts"]["periodic"]) == 1
    assert int(round_summaries[0]["prioritized_family_counts"]["periodic"]) == 1


def test_run_closure_search_pass_atomized_rows_do_not_spend_core_exact_budget(monkeypatch):
    monkeypatch.setenv("NESTY_ATOMIZED_LINEAR_SPAN_EXACT_QUOTA", "2")
    scored = []

    def _row(key, *, atomized, probe):
        return {
            "scaffold_id": key,
            "expr": ("var", 0),
            "child_key": key,
            "proposal_key": key,
            "local_probe_mse": probe,
            "local_fit_mse": probe,
            "candidate_child_size": 1,
            "scaffold_family": "affine",
            "atomized_linear_span": bool(atomized),
        }

    def _run_closure_search_pass_impl(**_kwargs):
        return {
            "stats": {"scaffolds_enumerated": 4},
            "candidate_rows": [
                _row("atomized_0", atomized=True, probe=0.01),
                _row("atomized_1", atomized=True, probe=0.02),
                _row("core_0", atomized=False, probe=0.10),
                _row("core_1", atomized=False, probe=0.11),
            ],
        }

    def _score_external_candidate_expr_fn(expr, **kwargs):
        scored.append(str(kwargs["candidate_meta"]["proposal_key"]))
        return {
            "expr": expr,
            "raw_mse": 1.0,
            "fit_loss": 1.0,
            "probe_loss": 1.0,
        }

    kwargs = _base_closure_search_kwargs()
    kwargs.update(
        {
            "closure_search_exact_topk": 2,
            "closure_search_emergent_aux_atoms_enable": True,
            "closure_search_pair_rescue_enable": False,
            "run_closure_search_pass_impl": _run_closure_search_pass_impl,
            "score_external_candidate_expr_fn": _score_external_candidate_expr_fn,
            "score_native_candidate_basis_state_fn": None,
        }
    )

    run_closure_search_pass(**kwargs)

    stats = kwargs["closure_search_stats"]
    assert scored == ["atomized_0", "atomized_1", "core_0", "core_1"]
    assert int(stats["atomized_linear_span_exact_scored"]) == 2


def test_run_closure_search_pass_pair_rescue_prefers_cross_family_pair_pool():
    base_state = basis_state_from_closure_candidate(
        family="affine",
        scaffold_id="affine:base",
        expr=("var", 3),
        anchor_node=None,
        scaffold_metadata={"form": "affine_base"},
        local_fit_mse=1.0,
        local_probe_mse=1.0,
        local_mapping_kind="direct_linear_head",
        direct_metadata={"feature_node": ("var", 3)},
    )
    state_periodic_a = basis_state_from_closure_candidate(
        family="periodic",
        scaffold_id="periodic:a",
        expr=("cos", ("var", 0)),
        anchor_node=None,
        scaffold_metadata={"form": "cos_base"},
        local_fit_mse=1.0,
        local_probe_mse=1.0,
        local_mapping_kind="direct_linear_head",
        direct_metadata={"feature_node": ("cos", ("var", 0)), "execution_mode": "exact_bound"},
    )
    state_periodic_b = basis_state_from_closure_candidate(
        family="periodic",
        scaffold_id="periodic:b",
        expr=("sin", ("var", 0)),
        anchor_node=None,
        scaffold_metadata={"form": "sin_base"},
        local_fit_mse=1.0,
        local_probe_mse=1.0,
        local_mapping_kind="direct_linear_head",
        direct_metadata={"feature_node": ("sin", ("var", 0)), "execution_mode": "exact_bound"},
    )
    state_power = basis_state_from_closure_candidate(
        family="power",
        scaffold_id="power:c",
        expr=("sqrt", ("var", 1)),
        anchor_node=None,
        scaffold_metadata={"form": "power_sqrt"},
        local_fit_mse=1.0,
        local_probe_mse=1.0,
        local_mapping_kind="direct_power_head",
        direct_metadata={"power_inner_node": ("var", 1), "execution_mode": "exact_bound"},
    )
    scored = []

    def _run_closure_search_pass_impl(**_kwargs):
        return {
            "stats": {},
            "candidate_rows": [
                {
                    "scaffold_id": "p1",
                    "expr": ("cos", ("var", 0)),
                    "child_key": "p1",
                    "proposal_key": "p1",
                    "local_probe_mse": 0.10,
                    "local_fit_mse": 0.10,
                    "candidate_child_size": 2,
                    "basis_state_obj": state_periodic_a,
                    "feature_block_obj": state_periodic_a.blocks[0],
                    "scaffold_family": "periodic",
                    "execution_mode": "exact_bound",
                },
                {
                    "scaffold_id": "p2",
                    "expr": ("sin", ("var", 0)),
                    "child_key": "p2",
                    "proposal_key": "p2",
                    "local_probe_mse": 0.11,
                    "local_fit_mse": 0.11,
                    "candidate_child_size": 2,
                    "basis_state_obj": state_periodic_b,
                    "feature_block_obj": state_periodic_b.blocks[0],
                    "scaffold_family": "periodic",
                    "execution_mode": "exact_bound",
                },
                {
                    "scaffold_id": "q1",
                    "expr": ("sqrt", ("var", 1)),
                    "child_key": "q1",
                    "proposal_key": "q1",
                    "local_probe_mse": 0.12,
                    "local_fit_mse": 0.12,
                    "candidate_child_size": 2,
                    "basis_state_obj": state_power,
                    "feature_block_obj": state_power.blocks[0],
                    "scaffold_family": "power",
                    "execution_mode": "exact_bound",
                },
            ],
        }

    def _score_external_candidate_expr_fn(expr, **kwargs):
        proposal_key = str(kwargs["candidate_meta"].get("proposal_key", ""))
        scored.append(proposal_key)
        candidate_state = kwargs["candidate_meta"]["basis_state_obj"]
        raw_mse = 1.0e-6 if proposal_key == "pair_rescue::p1++q1" else 1.0
        return {
            "expr": expr,
            "raw_mse": raw_mse,
            "fit_loss": raw_mse,
            "probe_loss": raw_mse,
            "basis_state_obj": candidate_state,
        }

    kwargs = _base_closure_search_kwargs()
    kwargs.update(
        {
            "closure_search_exact_topk": 1,
            "closure_search_pair_rescue_enable": True,
            "closure_search_pair_rescue_topk": 2,
            "closure_search_pair_rescue_max_pairs": 1,
            "proposal_context": ProposalContext(
                basis_state=base_state,
                basis_state_beam=admit_basis_state_to_beam([], base_state, beam_width=4),
                total_budget=3,
            ),
            "run_closure_search_pass_impl": _run_closure_search_pass_impl,
            "score_external_candidate_expr_fn": _score_external_candidate_expr_fn,
            "score_native_candidate_basis_state_fn": None,
        }
    )

    run_closure_search_pass(**kwargs)

    stats = kwargs["closure_search_stats"]
    assert scored == ["p1", "pair_rescue::p1++q1"]
    assert int(stats["basis_state_pair_rescue_scored"]) == 1
    assert int(stats["basis_state_pair_rescue_accepted"]) == 1


def test_run_closure_search_pass_pair_rescue_keeps_distinct_specs_within_family():
    base_state = basis_state_from_closure_candidate(
        family="affine",
        scaffold_id="affine:base",
        expr=("var", 3),
        anchor_node=None,
        scaffold_metadata={"form": "affine_base"},
        local_fit_mse=1.0,
        local_probe_mse=1.0,
        local_mapping_kind="direct_linear_head",
        direct_metadata={"feature_node": ("var", 3)},
    )
    state_periodic = basis_state_from_closure_candidate(
        family="periodic",
        scaffold_id="periodic:p1",
        expr=("cos", ("var", 0)),
        anchor_node=None,
        scaffold_metadata={"form": "cos_base"},
        local_fit_mse=1.0,
        local_probe_mse=1.0,
        local_mapping_kind="direct_linear_head",
        direct_metadata={"feature_node": ("cos", ("var", 0)), "execution_mode": "exact_bound"},
    )
    state_power_sqrt = basis_state_from_closure_candidate(
        family="power",
        scaffold_id="power:q1",
        expr=("sqrt", ("var", 1)),
        anchor_node=None,
        scaffold_metadata={"form": "power_sqrt"},
        local_fit_mse=1.0,
        local_probe_mse=1.0,
        local_mapping_kind="direct_power_head",
        direct_metadata={"power_inner_node": ("var", 1), "execution_mode": "exact_bound"},
    )
    state_power_square = basis_state_from_closure_candidate(
        family="power",
        scaffold_id="power:q2",
        expr=("sqr", ("var", 1)),
        anchor_node=None,
        scaffold_metadata={"form": "power_sqr"},
        local_fit_mse=1.0,
        local_probe_mse=1.0,
        local_mapping_kind="direct_power_head",
        direct_metadata={"power_inner_node": ("var", 1), "execution_mode": "exact_bound"},
    )
    scored = []

    def _run_closure_search_pass_impl(**_kwargs):
        return {
            "stats": {},
            "candidate_rows": [
                {
                    "scaffold_id": "p1",
                    "expr": ("cos", ("var", 0)),
                    "child_key": "p1",
                    "proposal_key": "p1",
                    "local_probe_mse": 0.10,
                    "local_fit_mse": 0.10,
                    "candidate_child_size": 2,
                    "basis_state_obj": state_periodic,
                    "feature_block_obj": state_periodic.blocks[0],
                    "scaffold_family": "periodic",
                    "operator_id": "periodic:cos",
                    "execution_mode": "exact_bound",
                },
                {
                    "scaffold_id": "q1",
                    "expr": ("sqrt", ("var", 1)),
                    "child_key": "q1",
                    "proposal_key": "q1",
                    "local_probe_mse": 0.11,
                    "local_fit_mse": 0.11,
                    "candidate_child_size": 2,
                    "basis_state_obj": state_power_sqrt,
                    "feature_block_obj": state_power_sqrt.blocks[0],
                    "scaffold_family": "power",
                    "operator_id": "power:sqrt",
                    "execution_mode": "exact_bound",
                },
                {
                    "scaffold_id": "q2",
                    "expr": ("sqr", ("var", 1)),
                    "child_key": "q2",
                    "proposal_key": "q2",
                    "local_probe_mse": 0.12,
                    "local_fit_mse": 0.12,
                    "candidate_child_size": 2,
                    "basis_state_obj": state_power_square,
                    "feature_block_obj": state_power_square.blocks[0],
                    "scaffold_family": "power",
                    "operator_id": "power:sqr",
                    "execution_mode": "exact_bound",
                },
            ],
        }

    def _score_external_candidate_expr_fn(expr, **kwargs):
        proposal_key = str(kwargs["candidate_meta"].get("proposal_key", ""))
        scored.append(proposal_key)
        candidate_state = kwargs["candidate_meta"]["basis_state_obj"]
        raw_mse = 1.0e-6 if proposal_key == "pair_rescue::p1++q2" else 1.0
        return {
            "expr": expr,
            "raw_mse": raw_mse,
            "fit_loss": raw_mse,
            "probe_loss": raw_mse,
            "basis_state_obj": candidate_state,
        }

    kwargs = _base_closure_search_kwargs()
    kwargs.update(
        {
            "closure_search_exact_topk": 1,
            "closure_search_pair_rescue_enable": True,
            "closure_search_pair_rescue_topk": 3,
            "closure_search_pair_rescue_max_pairs": 2,
            "proposal_context": ProposalContext(
                basis_state=base_state,
                basis_state_beam=admit_basis_state_to_beam([], base_state, beam_width=4),
                total_budget=3,
            ),
            "run_closure_search_pass_impl": _run_closure_search_pass_impl,
            "score_external_candidate_expr_fn": _score_external_candidate_expr_fn,
            "score_native_candidate_basis_state_fn": None,
        }
    )

    run_closure_search_pass(**kwargs)

    stats = kwargs["closure_search_stats"]
    assert scored == ["p1", "pair_rescue::p1++q1", "pair_rescue::p1++q2"]
    assert int(stats["basis_state_pair_rescue_scored"]) == 2
    assert int(stats["basis_state_pair_rescue_accepted"]) == 1


def test_score_external_candidate_expr_promotes_additive_basis_state():
    x = torch.tensor(
        [
            [0.4, 0.5, 0.6],
            [0.7, 0.8, 0.9],
        ],
        dtype=torch.float64,
    )
    y = (x[:, 0] * torch.sqrt(x[:, 1]) + x[:, 2]).unsqueeze(-1)
    expr = ("add", ("mul", ("var", 0), ("sqrt", ("var", 1))), ("var", 2))
    base_expr = ("mul", ("var", 0), ("sqrt", ("var", 1)))
    base_state = basis_state_from_closure_candidate(
        family="power",
        scaffold_id="power:sqrt",
        expr=base_expr,
        anchor_node=None,
        scaffold_metadata={"form": "sqrt"},
        local_fit_mse=1.0,
        local_probe_mse=1.0,
        local_mapping_kind="direct_linear_head",
        local_mapping_coeffs=[1.0, 0.0],
        direct_metadata={"feature_node": ("sqrt", ("var", 1))},
    )

    transition = make_additive_basis_transition(
        core_expr=base_expr,
        term_nodes=[("var", 2)],
        coeffs=[1.0, 1.0, 0.0],
        compiled_expr=expr,
        ridge=1.0e-8,
        prune_rel=1.0e-6,
    )

    class _Arch:
        def update(self, *args, **kwargs):
            return True

    def _score_expr_fn(_expr, *_args, **_kwargs):
        z = torch.zeros((int(y.shape[0]),), dtype=y.dtype)
        mapping = {
            "kind": "poly",
            "coeffs": [0.0, 1.0],
            "mu": 0.0,
            "std": 1.0,
            "_basis_transition": transition,
        }
        return 1.0e-20, ("expr", "demo"), z, mapping, expr

    stats = {}
    state = type(
        "_State",
        (),
        {"n_evaluated": 0, "best_raw_mse": float("inf"), "best_raw_mse_struct": float("inf"), "best_mse": float("inf")},
    )()
    scored = score_external_candidate_expr(
        expr,
        parent_raw_mse=None,
        stats=stats,
        route_name="closure_search",
        candidate_meta={
            "scaffold_family": "power",
            "scaffold_id": "power:sqrt",
            "basis_state_obj": base_state,
        },
        state=state,
        dm=False,
        var_dims=None,
        y_dims=None,
        refine_cfg={},
        score_prescreen_stats={},
        closure_search_anchor_head_compare_enable=False,
        x_fit=x,
        y_fit=y,
        x_probe=x,
        y_probe=y,
        proj=torch.zeros((int(y.shape[0]), 4), dtype=y.dtype),
        fp_mode="bits",
        q_scale=2.0,
        q_clip=6,
        poly_degree=2,
        refine_enable=False,
        refine_state=None,
        early_stop_mse=0.0,
        complexity_penalty=0.0,
        score_expr_fn=_score_expr_fn,
        simplify_fn=lambda node: node,
        is_valid_node_fn=lambda node: isinstance(node, tuple),
        node_str_fn=lambda node: str(node),
        node_dims_fn=lambda node, dims: None,
        dims_eq_fn=lambda a, b: a == b,
        node_size_fn=lambda node: 1,
        mapping_cost_fn=lambda mapping: 0.0,
        mapping_is_structural_fn=lambda mapping: True,
        arch=_Arch(),
    )

    assert scored is not None
    basis_state = scored["basis_state_obj"]
    assert basis_state is not None
    assert basis_state.to_dict()["block_count"] == 2
    assert basis_state.to_dict()["compiled_expr"] == "((sqrt(x1)*x0)+x2)"
    assert stats["basis_state_promotions"] == 1
