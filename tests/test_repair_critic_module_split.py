import inspect
import pickle

from nestynet_sr.sr_search.factorized_search import repair_critic as public_facade
from nestynet_sr.sr_search.factorized_search.legacy import repair_critic as legacy_facade


EXPECTED_GROUP_COUNTS = {
    "_FEATURE_FUNCTIONS": 46,
    "_MODEL_FUNCTIONS": 28,
    "_TRAINING_FUNCTIONS": 9,
    "_PREDICTION_FUNCTIONS": 12,
    "_MODEL_CLASSES": 5,
    "_CONSTANT_NAMES": 26,
}


def test_repair_critic_partition_and_historical_facades_are_complete():
    assert len(public_facade.__all__) == 146
    assert not (set(public_facade.__all__) & public_facade._split_internal_names)

    for group_name, expected_count in EXPECTED_GROUP_COUNTS.items():
        assert len(getattr(legacy_facade, group_name)) == expected_count

    function_groups = (
        (legacy_facade._features, legacy_facade._FEATURE_FUNCTIONS),
        (legacy_facade._models, legacy_facade._MODEL_FUNCTIONS),
        (legacy_facade._training, legacy_facade._TRAINING_FUNCTIONS),
        (legacy_facade._prediction, legacy_facade._PREDICTION_FUNCTIONS),
    )
    seen = set()
    for module, names in function_groups:
        assert seen.isdisjoint(names)
        seen.update(names)
        for name in names:
            implementation = getattr(module, name)
            legacy_symbol = getattr(legacy_facade, name)
            public_symbol = getattr(public_facade, name)
            assert legacy_symbol is implementation
            assert public_symbol is implementation
            assert inspect.signature(legacy_symbol) == inspect.signature(implementation)
            assert legacy_symbol.__module__ == (
                "nestynet_sr.sr_search.factorized_search.legacy.repair_critic"
            )
            assert legacy_symbol.__qualname__ == name
            assert pickle.loads(pickle.dumps(legacy_symbol)) is legacy_symbol

    assert len(seen) == 95

    for name in legacy_facade._MODEL_CLASSES:
        implementation = getattr(legacy_facade._models, name)
        assert getattr(legacy_facade, name) is implementation
        assert getattr(public_facade, name) is implementation
        assert implementation.__module__ == (
            "nestynet_sr.sr_search.factorized_search.legacy.repair_critic"
        )
        assert pickle.loads(pickle.dumps(implementation)) is implementation

    for name in legacy_facade._CONSTANT_NAMES:
        value = getattr(legacy_facade._features, name)
        assert getattr(legacy_facade, name) is value
        assert getattr(public_facade, name) is value


def test_repair_route_row_predictors_are_late_bound_without_cycle():
    assert legacy_facade._features.predict_build_tuple_slate is getattr(
        legacy_facade, "predict_build_tuple_slate"
    )
    assert legacy_facade._features.predict_repair_tuple_slate is getattr(
        legacy_facade, "predict_repair_tuple_slate"
    )


def test_repair_critic_private_modules_stay_below_review_threshold():
    modules = (
        legacy_facade._features,
        legacy_facade._models,
        legacy_facade._training,
        legacy_facade._prediction,
    )
    for module in modules:
        with open(module.__file__, encoding="utf-8") as handle:
            assert sum(1 for _ in handle) < 5_000
