import inspect
import pickle
import typing

from nestynet_sr.sr_search.factorized_search.engine import search as facade


EXPECTED_GROUPS = {
    "support": (23, 0, 6, 0),
    "state": (6, 5, 0, 0),
    "runtime": (6, 0, 0, 0),
}


def _type_hints_outcome(obj):
    try:
        return ("ok", typing.get_type_hints(obj))
    except Exception as exc:
        return ("error", type(exc), str(exc))


def test_engine_search_partition_and_facade_contract():
    assert facade.__all__ == ["Explorer", "run_explorer_core"]
    assert facade._log.name == "nestynet_sr.sr_search.factorized_search.engine.search"
    seen = set()
    constant_names = set()
    for module in facade._GROUP_MODULES:
        group = module.__name__.removeprefix(
            "nestynet_sr.sr_search.factorized_search.engine._search_"
        )
        names = module.__engine_search_definitions__
        constants = module.__engine_search_constants__
        late = module.__engine_search_late_bindings__
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
                assert inspect.signature(exported) == inspect.signature(implementation)
                assert inspect.getfullargspec(exported) == inspect.getfullargspec(implementation)
                assert exported.__defaults__ == implementation.__defaults__
                assert exported.__kwdefaults__ == implementation.__kwdefaults__
                assert exported.__code__.co_argcount == implementation.__code__.co_argcount
                assert exported.__code__.co_posonlyargcount == implementation.__code__.co_posonlyargcount
                assert exported.__code__.co_kwonlyargcount == implementation.__code__.co_kwonlyargcount
                assert _type_hints_outcome(exported) == _type_hints_outcome(implementation)
                assert inspect.unwrap(exported).__module__ == facade.__name__
                assert inspect.unwrap(exported).__qualname__ == name
            assert exported.__module__ == facade.__name__
            assert exported.__qualname__ == name
            assert pickle.loads(pickle.dumps(exported)) is exported

        for name in constants:
            expected = getattr(module, name)
            if name == "_log":
                assert facade._log.name.endswith("engine.search")
            else:
                assert getattr(facade, name) is expected

    assert len(seen) == 35
    assert len(constant_names) == 6


def test_engine_search_facade_monkeypatch_points_are_forwarded(monkeypatch):
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


def test_engine_runtime_hooks_bind_in_runtime_module(monkeypatch):
    hook_names = tuple(facade._LEGACY_SEARCH_HELPERS) + tuple(facade._OPTIONAL_RUNTIME_HOOKS)
    hooks = {name: object() for name in hook_names}
    with monkeypatch.context() as context:
        for name in hook_names:
            context.setattr(facade._runtime, name, object(), raising=False)
        facade._bind_runtime_hooks(hooks)
        for name, sentinel in hooks.items():
            assert getattr(facade._runtime, name) is sentinel


def test_engine_search_private_modules_stay_review_sized():
    for module in facade._GROUP_MODULES:
        with open(module.__file__, encoding="utf-8") as handle:
            lines = sum(1 for _ in handle)
        if module is facade._runtime:
            # The runtime contains one 7,853-line closure-heavy state machine.
            assert lines < 8_500
        else:
            assert lines < 5_000
