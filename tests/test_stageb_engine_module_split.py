from __future__ import annotations

import ast
import inspect
import pickle
from pathlib import Path

import nestynet_sr.sr_search.stageB._engine_runtime as runtime
import nestynet_sr.sr_search.stageB._engine_state as state
import nestynet_sr.sr_search.stageB._engine_support as support
import nestynet_sr.sr_search.stageB.engine as legacy


EXPECTED_DEFINITIONS = {
    "_engine_support": [
        "_snapshot_rng_state",
        "_restore_rng_state",
        "_safe_ast_cost",
        "_clamp_nonnegative_finite",
        "_loss_excess_above_floor",
        "_effective_loss_floor",
        "_best_seen_restore_decision",
        "_below_floor_regression_cap",
        "_below_floor_regression_rejected",
        "_candidate_mapping_cost",
        "_candidate_is_unpromoted_generic",
        "_mapping_descriptor",
        "_candidate_mapping_descriptor",
        "_candidate_has_mapping",
        "_candidate_is_structural_accept",
        "_phase2_trigger_flags",
        "_target_uid",
        "_eval_yspace_mse",
        "_asinh_yspace_scale_from_loader",
        "_loss_str",
        "_format_dim_for_problem",
        "_target_dim_for_root",
        "_input_basis_dims_for_atom",
        "_find_nonsense_units_leaves",
        "_annotate_nonsense_units_leaves",
        "_problem_candidate_desc",
        "candidate_pattern_name",
        "_count_ast_params",
        "_candidate_min_free_params",
        "_cand_sort_key",
        "_candidate_can_beat_floor_locked_state",
        "_is_exact_final_leaf_monomial_accept",
        "_stageB_state_num_params",
        "_stageB_state_num_nn_atoms",
        "_stageB_completion_loss_floor",
        "_min_following_candidate_free_params",
        "_are_we_done_yet",
        "_are_we_done_yet_reason",
        "_skip_post_accept_polish_for_terminal_state",
        "_count_effective_params",
        "_leaf_z_data",
        "_effective_ratpoly_params",
        "_effective_poly_params",
        "_unwrap_leaf_core",
        "_filter_reuse_map",
        "_find_ratpoly_scale_pair",
        "_ratpoly_degree_bands",
        "_ratpoly_support_degrees",
        "_format_ratpoly_support",
        "_ratpoly_den_pivot_degree",
        "_is_ratpoly_candidate",
        "_ratpoly_exps_key",
        "_ratpoly_support_signature_exact",
        "_refresh_ratpoly_trim_unit_certificate",
        "_ratpoly_num_pivot_degree",
        "_lookup_rratpoly_trim_target",
        "_lookup_ratpoly_trim_target",
        "_build_rratpoly_degree_trim_candidate",
        "_ast_node_to_tuple",
        "_target_arity",
        "atom_content_hash",
        "_is_structural_candidate",
        "_is_separability_candidate",
        "_nn_multivar_complexity",
        "_compute_nn_metrics"
    ],
    "_engine_state": [
        "StageBRule",
        "StageBState",
        "_Checkpoint",
        "_materialized_fit_state_for_checkpoint",
        "_checkpoint_state_dict_cpu",
        "_is_transient_fit_state_key",
        "_state_value_clone",
        "_load_checkpoint_state_dict",
        "Candidate",
        "PrecheckResult",
        "StageBContext"
    ],
    "_engine_runtime": [
        "_find_worst_accept",
        "_pick_atom_factory",
        "_restore_from_checkpoint",
        "StageBEngine"
    ]
}
EXPECTED_CONSTANTS = {
    "_engine_support": [
        "GREEN",
        "PURPLE",
        "RED",
        "RESET",
        "GAUGE_SCOPE_RULES",
        "GAUGE_TERMINALISH_RULES",
        "GAUGE_SENSITIVE_RULES",
        "STRUCTURAL_LABEL_PREFIXES",
        "STRUCTURAL_LABELS",
        "SEPARABILITY_LABELS"
    ],
    "_engine_state": [
        "_TRANSIENT_FIT_STATE_SUFFIXES"
    ],
    "_engine_runtime": []
}
EXPECTED_PUBLIC_NAMES = [
    "annotations",
    "copy",
    "math",
    "random",
    "time",
    "dataclass",
    "field",
    "replace",
    "groupby",
    "Any",
    "Callable",
    "Dict",
    "List",
    "Optional",
    "Set",
    "Tuple",
    "torch",
    "AddNode",
    "AtomNode",
    "MulNode",
    "PowNode",
    "LogNode",
    "ExpNode",
    "SinNode",
    "CosNode",
    "AsinNode",
    "AcosNode",
    "AtanNode",
    "ConjNode",
    "RealNode",
    "ImagNode",
    "AbsNode",
    "ArgNode",
    "Node",
    "collect_all_atoms",
    "collect_nn_atoms",
    "atom_problem_label",
    "count_atom_params",
    "effective_arity",
    "eval_inputs",
    "get_input_exprs",
    "clone_ast",
    "clone_inputs",
    "ast_to_human_readable",
    "UnitsSpec",
    "check_units_ast",
    "compute_node_domains",
    "eval_analytic_expr_dim",
    "is_dimless",
    "scale_dim",
    "infer_atom_output_dim",
    "CoEWitnessExecutor",
    "coe_stageB_refit_ast_to_payload",
    "coe_witness_execution_metadata",
    "coe_witness_jobs_from_specs",
    "run_fixed_expression_pair_witnesses",
    "run_stageB_refit_pair_witnesses",
    "run_stageB_refit_pair_witness_preflight",
    "summarize_witness_errors",
    "candidate_monomial_exponent",
    "AdditiveGaugeGlobalScore",
    "AdditiveGaugeScopeIndex",
    "additive_gauge_global_score",
    "HomogeneousGaugeGlobalScore",
    "HomogeneousGaugeScopeIndex",
    "homogeneous_gauge_global_score",
    "GREEN",
    "PURPLE",
    "RED",
    "RESET",
    "GAUGE_SCOPE_RULES",
    "GAUGE_TERMINALISH_RULES",
    "GAUGE_SENSITIVE_RULES",
    "STRUCTURAL_LABEL_PREFIXES",
    "STRUCTURAL_LABELS",
    "candidate_pattern_name",
    "SEPARABILITY_LABELS",
    "atom_content_hash",
    "StageBRule",
    "StageBState",
    "Candidate",
    "PrecheckResult",
    "StageBContext",
    "StageBEngine"
]
MODULES = {
    "_engine_support": support,
    "_engine_state": state,
    "_engine_runtime": runtime,
}
EXPECTED_INTERNAL_DEFINITIONS = {"_refresh_ratpoly_trim_unit_certificate"}


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


def test_stageb_engine_split_is_complete_and_disjoint():
    observed = []
    for module_name, module in MODULES.items():
        definitions, constants = _source_inventory(module)
        assert definitions == EXPECTED_DEFINITIONS[module_name]
        assert constants == EXPECTED_CONSTANTS[module_name]
        observed.extend(definitions)
    assert len(observed) == 80
    assert len(set(observed)) == 80
    assert set(observed) == (
        set(legacy._HISTORICAL_DEFINITIONS) | EXPECTED_INTERNAL_DEFINITIONS
    )


def test_stageb_engine_preserves_historical_surface_and_reflection():
    assert not hasattr(legacy, "__all__")
    namespace = {}
    exec("from nestynet_sr.sr_search.stageB.engine import *", {}, namespace)
    observed_public = [name for name in namespace if not name.startswith("_")]
    assert observed_public == EXPECTED_PUBLIC_NAMES

    for name in legacy._HISTORICAL_DEFINITIONS:
        obj = getattr(legacy, name)
        assert obj.__module__ == legacy.__name__
        assert pickle.loads(pickle.dumps(obj)) is obj
        if inspect.isclass(obj):
            for member in vars(obj).values():
                targets = []
                if isinstance(member, (staticmethod, classmethod)):
                    targets.append(member.__func__)
                elif isinstance(member, property):
                    targets.extend(
                        target
                        for target in (member.fget, member.fset, member.fdel)
                        if target is not None
                    )
                elif inspect.isfunction(member):
                    targets.append(member)
                assert all(target.__module__ == legacy.__name__ for target in targets)


def test_stageb_engine_propagates_historical_monkeypatches(monkeypatch):
    sentinels = {
        name: object()
        for name in (
            "_lookup_rratpoly_trim_target",
            "_lookup_ratpoly_trim_target",
            "_build_rratpoly_degree_trim_candidate",
        )
    }
    for name, sentinel in sentinels.items():
        monkeypatch.setattr(legacy, name, sentinel)
        assert getattr(support, name) is sentinel
        assert getattr(state, name) is sentinel
        assert getattr(runtime, name) is sentinel


def test_stageb_engine_modules_are_review_sized():
    assert len(Path(legacy.__file__).read_text().splitlines()) < 500
    for module in MODULES.values():
        assert len(Path(module.__file__).read_text().splitlines()) < 5000
