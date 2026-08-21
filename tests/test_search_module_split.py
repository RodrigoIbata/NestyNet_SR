import inspect
import pickle
import typing

import nestynet_sr.sr_search.search as facade


EXPECTED_GROUPS = {
    "shadow": (37, 0, 7, 4),
    "training": (41, 1, 0, 0),
    "detection": (11, 0, 0, 13),
    "structure": (41, 0, 4, 11),
    "proposals": (75, 0, 1, 4),
    "compounds": (7, 0, 0, 6),
    "policy": (32, 2, 0, 1),
    "runtime": (1, 0, 0, 0),
}

EXPECTED_PUBLIC_NAMES = [
    "copy",
    "itertools",
    "json",
    "math",
    "random",
    "dataclass",
    "Fraction",
    "Any",
    "Dict",
    "Iterable",
    "List",
    "Optional",
    "Tuple",
    "torch",
    "DataLoader",
    "Dataset",
    "TensorDataset",
    "Var",
    "ast_to_human_readable",
    "build_linear_ast",
    "build_mixed_compound_ast",
    "build_monomial_ast",
    "build_radial_r2_ast",
    "check_linear_compound",
    "check_mixed_compound",
    "check_monomial_compound",
    "check_monomial_compound_logderiv",
    "check_separability_ops",
    "collect_nn_atoms",
    "replace_atom_in_ast",
    "AcosNode",
    "AddNode",
    "AsinNode",
    "AtanNode",
    "AtomNode",
    "ConstNode",
    "CosNode",
    "ExpNode",
    "LogNode",
    "MulNode",
    "Node",
    "PowNode",
    "Scale",
    "SinNode",
    "ast_equals",
    "clone_ast",
    "compound_input_expr",
    "effective_arity",
    "eval_input_expr",
    "eval_inputs",
    "extra_input_var_idxs",
    "get_input_exprs",
    "has_nontrivial_input",
    "is_trivial_input",
    "canonical_fit_link_name",
    "describe_fit_link",
    "fit_link_torch",
    "CoEWitnessExecutor",
    "coe_witness_execution_metadata",
    "coe_witness_jobs_from_specs",
    "run_threaded_witnesses",
    "LeafFeatures",
    "TrigAxisSpec",
    "TrigProbeTarget",
    "TrigScaleSpec",
    "discover_compound_features_from_data",
    "discover_constant_directions",
    "discover_invariance_features",
    "discover_leaf_features",
    "discover_model_directions",
    "discover_parity_axes",
    "discover_poly_in_f2",
    "discover_poly_in_x",
    "discover_preferred_origins",
    "discover_radial_groups",
    "discover_rational_poly",
    "discover_saturating_axes",
    "discover_scaling_features",
    "discover_trig_axes",
    "poisson_profile",
    "probe_oracle_scaling",
    "probe_trig_scaling",
    "sample_line_curvature",
    "trig_from_profile",
    "verify_compound_null_test",
    "candidate_monomial_exponent",
    "candidate_priority_from_screen",
    "fit_univariate_monomial_screen",
    "half_power_domain_ok",
    "monomial_power_label",
    "snap_to_half_integer_monomial_power",
    "clean_subset_patterns",
    "expand_forced_power_vector",
    "R1OperatorCertificate",
    "build_r1_certificate_replacement",
    "r1_certificate_poly_init",
    "scan_r1_operator_certificates",
    "ShadowCoordinate",
    "ShadowRegistry",
    "shadow_parent_key",
    "build_barycentric_compound_proposals",
    "build_logexp_compound_proposals",
    "build_metric_distance_compound_proposals",
    "stageA_tuple_from_proposal",
    "build_composite_ast",
    "is_minimal_ast",
    "train_candidate_model",
    "train_initial_model",
    "build_compound_z_variants",
    "compound_z_wrapper_policy",
    "should_select_compound_variant",
    "snap_omega",
    "precision_for_transform",
    "RED",
    "GREEN",
    "YELLOW",
    "BLUE",
    "PURPLE",
    "RESET",
    "StageAOut",
    "SplitPlan",
    "stageA_analyze",
    "run_separability_for_transform",
]


def _type_hints_outcome(obj):
    try:
        return ("ok", typing.get_type_hints(obj))
    except Exception as exc:
        return ("error", type(exc), str(exc))


def test_search_partition_and_facade_contract():
    assert [name for name in vars(facade) if not name.startswith("_")] == EXPECTED_PUBLIC_NAMES
    seen = set()
    constant_names = set()
    for module in facade._GROUP_MODULES:
        group = module.__name__.removeprefix("nestynet_sr.sr_search._search_")
        names = module.__search_definitions__
        constants = module.__search_constants__
        late = module.__search_late_bindings__
        n_classes = sum(inspect.isclass(facade._IMPLEMENTATIONS[name]) for name in names)
        assert (len(names), n_classes, len(constants), len(late)) == EXPECTED_GROUPS[group]
        assert seen.isdisjoint(names)
        assert constant_names.isdisjoint(constants)
        seen.update(names)
        constant_names.update(constants)

        for name in names:
            implementation = facade._IMPLEMENTATIONS[name]
            exported = getattr(facade, name)
            if inspect.isclass(implementation):
                assert exported is implementation
            else:
                assert exported.__wrapped__ is implementation
                assert inspect.unwrap(exported) is inspect.unwrap(implementation)
                assert inspect.signature(exported) == inspect.signature(implementation)
                assert inspect.getfullargspec(exported) == inspect.getfullargspec(implementation)
                assert exported.__defaults__ == implementation.__defaults__
                assert exported.__kwdefaults__ == implementation.__kwdefaults__
                assert exported.__code__.co_argcount == implementation.__code__.co_argcount
                assert exported.__code__.co_posonlyargcount == implementation.__code__.co_posonlyargcount
                assert exported.__code__.co_kwonlyargcount == implementation.__code__.co_kwonlyargcount
                assert _type_hints_outcome(exported) == _type_hints_outcome(implementation)
            assert exported.__module__ == "nestynet_sr.sr_search.search"
            assert exported.__qualname__ == name
            assert pickle.loads(pickle.dumps(exported)) is exported

        for name in constants:
            assert getattr(facade, name) is getattr(module, name)

        for name in late:
            assert getattr(module, name) is getattr(facade, name)

    assert len(seen) == 245
    assert len(constant_names) == 12


def test_search_facade_monkeypatch_points_are_forwarded(monkeypatch):
    sentinels = {name: object() for name in facade._PATCHABLE_GLOBAL_NAMES}
    with monkeypatch.context() as context:
        for name, sentinel in sentinels.items():
            context.setattr(facade, name, sentinel)
        facade._sync_patchable_globals()
        for module in facade._GROUP_MODULES:
            for name, sentinel in sentinels.items():
                if hasattr(module, name):
                    assert getattr(module, name) is sentinel

    facade._sync_patchable_globals()


def test_search_private_modules_stay_review_sized():
    for module in facade._GROUP_MODULES:
        with open(module.__file__, encoding="utf-8") as handle:
            lines = sum(1 for _ in handle)
        if module is facade._runtime:
            # This is one closure-heavy state-machine function preserved intact.
            assert lines < 7_500
        else:
            assert lines < 5_000
