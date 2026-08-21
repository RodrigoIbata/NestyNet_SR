import ast
import inspect
import pickle
from collections import Counter
from pathlib import Path

import nestynet_sr.sr_search.factorized_search.explorer as legacy
from nestynet_sr.sr_search.factorized_search import _explorer_actions as actions
from nestynet_sr.sr_search.factorized_search import _explorer_brute as brute
from nestynet_sr.sr_search.factorized_search import _explorer_scoring as scoring


EXPECTED_DEFINITIONS = {
    "actions": [
        "_use_affine_fast_path",
        "_fit_best_with_cfg",
        "pb011_function",
        "addsum_function",
        "poly_function",
        "exp_product",
        "square_addsum",
        "feynman_012",
        "feynman_090",
        "feynman_028",
        "_coerce_guided_path",
        "_action_candidate_paths",
        "_select_action_path",
        "_normalize_controller_build_slate_actions",
        "_collect_controller_build_slate",
        "_controller_selected_action_path",
        "_score_repair_option_expr",
        "_eval_mapping_total_local",
        "_tracked_macro_actions",
        "_macro_action_fields",
        "_macro_decision_log_fields",
        "_merge_inverse_proposal_log_fields",
        "_merge_repair_option_log_fields",
        "apply_action",
        "apply_crossover_action",
        "apply_residual_action",
        "apply_inverse_steering_action",
        "run_repair_option"
    ],
    "scoring": [
        "_balanced_add_tree",
        "_strip_scalar_prefix",
        "_extract_scalar_core",
        "_collect_linear_terms",
        "_mapping_equiv_root",
        "_compile_linear_combo",
        "_harvest_pool_from_archive",
        "apply_boost_action",
        "fingerprint",
        "_negate_smart",
        "_pick_best_equiv_score",
        "_score_expr_base",
        "_score_expr_base_joint_affine",
        "_score_expr_base_joint_linear_terms",
        "_collect_trig_paths",
        "_trig_arg_has_const_scale",
        "_collect_log_paths",
        "_log_arg_has_const_scale",
        "_collect_exp_paths",
        "_exp_arg_has_const_scale",
        "_collect_sqr_shift_paths",
        "_sqr_shift_already_present",
        "_collect_sqrt_shift_paths",
        "_sqrt_shift_already_present",
        "_wrap_param_slots",
        "_refine_diag",
        "_diag_inc",
        "_diag_inc_context",
        "_diag_add_time",
        "_node_var_indices",
        "_refine_tensor_signature",
        "_refine_cfg_signature",
        "_refine_attempt_cache_key",
        "_refine_cache_get",
        "_refine_cache_put",
        "_prune_mapping_equiv_root_slot_paths",
        "_decorate_refine_variants",
        "_eval_node_hparam",
        "_materialize_hparams",
        "_build_init_logs",
        "_raw_to_hparams",
        "_stable_seed_from_text",
        "_select_subset_indices",
        "_slice_fit_subset",
        "_slice_fit_subset_multi",
        "_solve_linear_coeffs",
        "_solve_linearized_fit",
        "_joint_dataset_weights",
        "_solve_linearized_fit_multi",
        "_linearized_loss_value",
        "_build_single_slot_variant",
        "_slot_sensitivity_score",
        "_rank_paths_by_sensitivity",
        "_variant_has_gate_potential",
        "_build_grid_seed_logs",
        "_normalize_refine_optimizer",
        "_score_refine_raw_log",
        "_ranked_grid_refine_seeds",
        "_init_logs_from_grid_rank",
        "_flatten_add_terms",
        "_select_linear_basis_nodes",
        "_eval_node_hparam_safe",
        "_build_phi_hparam",
        "_materialize_linearized_candidate",
        "_refine_hparams",
        "_refine_budget_left",
        "score_expr"
    ],
    "brute": [
        "_init_crossover_policy_stats",
        "_finalize_crossover_policy_stats",
        "_remove_allowed_action",
        "_finalize_action_distribution",
        "enumerate_trees",
        "enumerate_trees_dim",
        "_has_const_zero",
        "_dedup_new",
        "_auto_brute_depth",
        "_enumerate_incremental",
        "_enumerate_dim_incremental",
        "_build_monomial_ast",
        "_monomial_presearch",
        "_lorentz_peel_presearch",
        "_planck_peel_presearch",
        "_hyperbolic_peel_presearch",
        "_gaussian_peel_presearch",
        "_invtrig_peel_presearch",
        "_archive_best_mse",
        "_archive_best_structural_mse",
        "_promote_structural_shadow_archive",
        "_run_brute_phase"
    ],
    "facade": [
        "make_engine_refinement_hooks",
        "make_engine_runtime_hooks",
        "score_expr",
        "run_explorer_core",
        "main"
    ]
}
EXPECTED_CONSTANTS = {
    "actions": [
        "TARGET_FUNCS",
        "A_REPLACE",
        "A_WRAP_UNARY",
        "A_ADD_RAND",
        "A_MUL_RAND",
        "A_RESIDUAL",
        "A_PRUNE",
        "A_CROSSOVER",
        "A_BOOST",
        "A_INVSTEER",
        "A_REPAIR",
        "A_HOLESEARCH",
        "A_CROSSOVER_LOCAL",
        "A_CROSSOVER_FOREIGN",
        "ACTIONS",
        "ACTION_NAME",
        "ACTION_ID_BY_NAME",
        "_INVERSE_CANDIDATE_META_KEYS",
        "_INVERSE_EXTRA_META_KEYS",
        "_CONTROLLER_BUILD_SLATE_DEFAULT_ACTIONS"
    ],
    "scoring": [],
    "brute": [],
    "facade": [
        "_log",
        "_mapping_equiv_root",
        "_harvest_pool_from_archive",
        "_eval_node_hparam_safe"
    ]
}
EXPECTED_PUBLIC_NAMES = [
    "argparse",
    "math",
    "random",
    "json",
    "hashlib",
    "time",
    "itertools",
    "Any",
    "Mapping",
    "Sequence",
    "torch",
    "make_additive_basis_transition",
    "InverseSteeringConfig",
    "coerce_inverse_steering_config",
    "mapping_cost",
    "ResidualBasinArchive",
    "Elite",
    "Rec",
    "CandidateStateFeatures",
    "InverseSteeringPotential",
    "PathStateFeatures",
    "build_controller_state_record",
    "coerce_repair_feature_row",
    "RepairControllerFeatureRecord",
    "choose_parent",
    "choose_parent_repair_aware",
    "load_repair_critic_bundle",
    "predict_repair_build_route",
    "predict_repair_controller_heads",
    "load_opportunity_bundle",
    "predict_opportunity_slate",
    "RESEARCH_PROFILE_NAMES",
    "resolve_engine_research_profile",
    "shared_candidate_row_dict",
    "MacroController",
    "build_macro_controller_state",
    "BINARY_OPS",
    "UNARY_OPS",
    "build_pool",
    "cap_depth",
    "collect_paths",
    "compute_reachable",
    "dim_round",
    "dims_eq",
    "eval_node",
    "get_at",
    "node_cost_physics_prior",
    "node_depth",
    "node_dims",
    "node_size",
    "node_str",
    "rand_node",
    "rand_node_dim",
    "replace_at",
    "sample_box",
    "set_dim_precision",
    "simplify",
    "eval_exp_mapping",
    "eval_mapping",
    "eval_pade",
    "eval_poly",
    "eval_power",
    "eval_sine",
    "fit_best",
    "fit_exp_mapping",
    "fit_pade",
    "fit_poly",
    "fit_power",
    "fit_sine",
    "mean_squared_error_same_shape",
    "mapping_is_structural",
    "InverseStep",
    "InverseTarget",
    "eval_mapping_total",
    "invert_context_target",
    "invert_context_target_beam",
    "invert_mapping_target",
    "estimate_inverse_steering_potential",
    "run_inverse_steering_action",
    "run_repair_option_action",
    "pb011_function",
    "addsum_function",
    "poly_function",
    "exp_product",
    "square_addsum",
    "feynman_012",
    "feynman_090",
    "feynman_028",
    "TARGET_FUNCS",
    "A_REPLACE",
    "A_WRAP_UNARY",
    "A_ADD_RAND",
    "A_MUL_RAND",
    "A_RESIDUAL",
    "A_PRUNE",
    "A_CROSSOVER",
    "A_BOOST",
    "A_INVSTEER",
    "A_REPAIR",
    "A_HOLESEARCH",
    "A_CROSSOVER_LOCAL",
    "A_CROSSOVER_FOREIGN",
    "ACTIONS",
    "ACTION_NAME",
    "ACTION_ID_BY_NAME",
    "apply_action",
    "apply_crossover_action",
    "apply_residual_action",
    "apply_inverse_steering_action",
    "run_repair_option",
    "apply_boost_action",
    "fingerprint",
    "score_expr",
    "Explorer",
    "enumerate_trees",
    "enumerate_trees_dim",
    "make_engine_refinement_hooks",
    "make_engine_runtime_hooks",
    "run_explorer_core",
    "main"
]
MODULES = {
    "actions": actions,
    "scoring": scoring,
    "brute": brute,
}


def _source_inventory(module):
    tree = ast.parse(Path(module.__file__).read_text())
    definitions = [
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    ]
    constants = []
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        for target in targets:
            if (
                isinstance(target, ast.Name)
                and (target.id.isupper() or target.id.startswith("_"))
            ):
                constants.append(target.id)
    return definitions, constants


def test_explorer_split_is_complete_and_disjoint():
    observed = []
    for module_name, module in MODULES.items():
        definitions, constants = _source_inventory(module)
        assert definitions == EXPECTED_DEFINITIONS[module_name]
        assert constants == EXPECTED_CONSTANTS[module_name]
        observed.extend(definitions)

    facade_definitions, facade_constants = _source_inventory(legacy)
    assert facade_definitions == [
        *EXPECTED_DEFINITIONS["facade"],
        "_CompatibilityModule",
    ]
    assert facade_constants[: len(EXPECTED_CONSTANTS["facade"])] == EXPECTED_CONSTANTS["facade"]
    observed.extend(EXPECTED_DEFINITIONS["facade"])

    assert len(observed) == 122
    assert len(set(observed)) == 121
    assert Counter(observed)["score_expr"] == 2


def test_explorer_preserves_historical_surface_and_reflection():
    assert not hasattr(legacy, "__all__")
    namespace = {}
    exec(
        "from nestynet_sr.sr_search.factorized_search.explorer import *",
        {},
        namespace,
    )
    observed_public = [name for name in namespace if not name.startswith("_")]
    assert observed_public == EXPECTED_PUBLIC_NAMES

    for name in dict.fromkeys(
        name
        for group in EXPECTED_DEFINITIONS.values()
        for name in group
    ):
        obj = getattr(legacy, name)
        expected_module = (
            "nestynet_sr.sr_search.factorized_search.engine.scoring"
            if name in {
                "_mapping_equiv_root",
                "fingerprint",
                "_harvest_pool_from_archive",
                "_eval_node_hparam_safe",
            }
            else legacy.__name__
        )
        unwrap_seen = set()
        unwrap_target = obj
        while inspect.isfunction(unwrap_target) and id(unwrap_target) not in unwrap_seen:
            unwrap_seen.add(id(unwrap_target))
            assert unwrap_target.__module__ == expected_module
            unwrap_target = getattr(unwrap_target, "__wrapped__", None)
        assert pickle.loads(pickle.dumps(obj)) is obj
        assert callable(inspect.unwrap(obj))


def test_explorer_propagates_historical_monkeypatches(monkeypatch):
    for name in (
        "rand_node",
        "apply_action",
        "score_expr",
        "_refine_hparams",
        "_run_brute_phase",
        "estimate_inverse_steering_potential",
    ):
        sentinel = object()
        monkeypatch.setattr(legacy, name, sentinel)
        assert getattr(actions, name) is sentinel
        assert getattr(scoring, name) is sentinel
        assert getattr(brute, name) is sentinel


def test_explorer_modules_are_review_sized():
    assert len(Path(legacy.__file__).read_text().splitlines()) < 1500
    for module in MODULES.values():
        assert len(Path(module.__file__).read_text().splitlines()) < 5000
