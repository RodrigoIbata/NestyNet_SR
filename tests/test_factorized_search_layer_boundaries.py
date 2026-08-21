# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

from __future__ import annotations

import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _imported_names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module.endswith("explorer") or module == "explorer":
                for alias in node.names:
                    names.add(alias.name)
    return names


def _loads_run_explorer_core_via_engine_search(path: Path) -> bool:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom):
            continue
        module = node.module or ""
        if not module.endswith("engine.search"):
            continue
        for alias in node.names:
            if alias.name == "run_explorer_core":
                return True
    return False


def test_direct_engine_consumers_import_search_entry_from_engine_surface():
    rel_paths = [
        "nestynet_sr/sr_search/factorized_search/bridge.py",
        "nestynet_sr/sr_search/factorized_search/oracle_lab.py",
        "nestynet_sr/sr_search/factorized_search/oracle_lab_de.py",
        "nestynet_sr/sr_search/factorized_search/opportunity_benchmark.py",
        "nestynet_sr/sr_search/factorized_search/controller_harness.py",
        "nestynet_sr/sr_search/factorized_search/stage1_benchmark_harness.py",
        "nestynet_sr/sr_search/factorized_search/repair_slate_report_gen.py",
    ]

    for rel_path in rel_paths:
        path = REPO_ROOT / rel_path
        imported = _imported_names(path)
        assert "run_explorer_core" not in imported, rel_path
        assert _loads_run_explorer_core_via_engine_search(path), rel_path


def test_engine_scoring_legacy_bridge_is_refinement_only():
    from nestynet_sr.sr_search.factorized_search.engine import scoring

    assert set(scoring._LEGACY_REFINEMENT_HELPERS) == {
        "_decorate_refine_variants",
        "_materialize_linearized_candidate",
        "_refine_hparams",
        "_variant_has_gate_potential",
    }


def test_engine_search_legacy_bridge_excludes_extracted_primitives():
    from nestynet_sr.sr_search.factorized_search.engine import search

    forbidden = {
        "ACTION_ID_BY_NAME",
        "ACTION_NAME",
        "A_ADD_RAND",
        "A_BOOST",
        "A_CROSSOVER",
        "A_HOLESEARCH",
        "A_INVSTEER",
        "A_MUL_RAND",
        "A_PRUNE",
        "A_REPAIR",
        "A_REPLACE",
        "A_RESIDUAL",
        "A_WRAP_UNARY",
        "_actor_critic_reward_terms",
        "_analytic_repair_controller_score",
        "_collect_controller_build_slate",
        "_controller_selected_action_path",
        "_finalize_action_distribution",
        "_finalize_crossover_policy_stats",
        "_hybrid_repair_controller_scores",
        "_init_crossover_policy_stats",
        "_macro_action_fields",
        "_macro_decision_log_fields",
        "_merge_inverse_proposal_log_fields",
        "_merge_repair_option_log_fields",
        "_normalize_repair_controller_critic_mode",
        "_remove_allowed_action",
        "_repair_controller_component_gate",
        "_repair_controller_path_policy",
        "_repair_controller_stagnation_state",
        "_repair_controller_threshold",
        "_repair_controller_weights",
        "_repair_option_candidate_paths",
        "_repair_parent_preview_retry_gate",
        "_repair_parent_record_attempt",
        "_repair_parent_retry_gate",
        "_repair_preview_signature",
        "_repair_route_compare_decision",
        "_select_action_path",
        "_tracked_macro_actions",
        "apply_action",
        "apply_crossover_action",
        "apply_residual_action",
        "ResidualBasinArchive",
        "InverseSteeringPotential",
        "MacroController",
        "RepairControllerFeatureRecord",
        "build_pool",
        "choose_parent",
        "choose_parent_repair_aware",
        "compute_reachable",
        "dim_round",
        "dims_eq",
        "estimate_inverse_steering_potential",
        "eval_mapping_total",
        "eval_node",
        "load_repair_critic_bundle",
        "mapping_cost",
        "mapping_is_structural",
        "node_depth",
        "node_dims",
        "node_size",
        "node_str",
        "rand_node",
        "rand_node_dim",
        "predict_repair_build_route",
        "predict_repair_controller_heads",
        "sample_box",
        "set_dim_precision",
        "simplify",
    }
    assert forbidden.isdisjoint(set(search._LEGACY_SEARCH_HELPERS))


def test_engine_modules_do_not_import_explorer_sideways():
    engine_paths = [
        REPO_ROOT / "nestynet_sr/sr_search/factorized_search/engine/search.py",
        REPO_ROOT / "nestynet_sr/sr_search/factorized_search/engine/scoring.py",
    ]
    for path in engine_paths:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                assert not module.endswith("explorer"), str(path)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id != "import_module":
                    continue
                assert not (
                    node.args
                    and isinstance(node.args[0], ast.Constant)
                    and str(node.args[0].value).endswith("explorer")
                ), str(path)
