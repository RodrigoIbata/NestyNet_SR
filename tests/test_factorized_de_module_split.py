import inspect
import pickle

from nestynet_sr.sr_de import factorized_de as facade


EXPECTED_GROUPS = {
    "frontend": (83, 3, 7, 5),
    "search": (35, 0, 0, 1),
    "operator": (103, 6, 6, 2),
    "lanes": (15, 0, 1, 0),
    "explorer": (20, 0, 0, 0),
    "rescue": (11, 0, 0, 0),
}

EXPECTED_PUBLIC_ALL = [
    "DEFeatureGroup",
    "FactorizedSearchDERescueConfig",
    "FactorizedSearchDEResult",
    "de_lab_spec_from_de_cfg",
    "default_physics_rescue_hp",
    "run_factorized_de_from_feature_groups",
    "run_direct_residual_fss_from_feature_groups",
    "run_regularized_implicit_residual_fss_from_feature_groups",
    "factorized_search_report_to_de_result",
    "normalized_rmse",
    "evaluate_factorized_search_candidate",
    "factorized_search_candidate_to_feature_predictor",
    "factorized_search_report_shortlist",
    "factorized_search_report_to_rhs_callable",
    "validate_order2_generator_witness",
    "build_factorized_search_de_feature_groups_from_surrogate",
    "build_factorized_search_de_feature_groups_from_surrogates",
    "run_factorized_search_de_from_feature_groups",
    "run_factorized_search_de_from_surrogate",
    "run_factorized_search_de_from_surrogates",
    "FactorizedDEBlock",
    "FactorizedDERescueConfig",
    "FactorizedDEResult",
    "run_factorized_coeff_rescue_from_feature_groups",
]


def test_factorized_de_partition_and_facade_contract():
    assert facade.__all__ == EXPECTED_PUBLIC_ALL
    seen = set()
    constant_names = set()
    for module in facade._GROUP_MODULES:
        group = module.__name__.rsplit("_", 1)[-1]
        names = module.__factorized_de_definitions__
        constants = module.__factorized_de_constants__
        late = module.__factorized_de_late_bindings__
        n_classes = sum(inspect.isclass(getattr(module, name)) for name in names)
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
                assert inspect.unwrap(exported) is implementation
                assert inspect.signature(exported) == inspect.signature(implementation)
                assert inspect.getfullargspec(exported) == inspect.getfullargspec(implementation)
                assert exported.__defaults__ == implementation.__defaults__
                assert exported.__kwdefaults__ == implementation.__kwdefaults__
            assert exported.__module__ == "nestynet_sr.sr_de.factorized_de"
            assert exported.__qualname__ == name
            assert pickle.loads(pickle.dumps(exported)) is exported

        for name in constants:
            assert getattr(facade, name) is getattr(module, name)

        for name in late:
            assert getattr(module, name) is getattr(facade, name)

    assert len(seen) == 267
    assert len(constant_names) == 14


def test_factorized_de_facade_monkeypatch_points_are_forwarded(monkeypatch):
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


def test_factorized_de_private_modules_stay_below_review_threshold():
    for module in facade._GROUP_MODULES:
        with open(module.__file__, encoding="utf-8") as handle:
            assert sum(1 for _ in handle) < 5_000
