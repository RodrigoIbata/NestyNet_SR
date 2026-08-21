import sys
import types


def _install_nestynet_adaptor_stubs():
    if "nestynet.adaptors.adaptors" in sys.modules:
        return
    nestynet_mod = types.ModuleType("nestynet")
    adaptors_pkg = types.ModuleType("nestynet.adaptors")
    adaptors_mod = types.ModuleType("nestynet.adaptors.adaptors")
    stacking_mod = types.ModuleType("nestynet.adaptors.stacking_adaptors")

    class _DummyAdaptor:
        pass

    adaptors_mod.AutogradAdaptor = _DummyAdaptor
    adaptors_mod.LAProvider = _DummyAdaptor
    adaptors_mod.SegmentedAdaptor = _DummyAdaptor
    stacking_mod.DualSegmentedAdaptor = _DummyAdaptor
    nestynet_mod.adaptors = adaptors_pkg
    adaptors_pkg.adaptors = adaptors_mod
    adaptors_pkg.stacking_adaptors = stacking_mod
    sys.modules["nestynet"] = nestynet_mod
    sys.modules["nestynet.adaptors"] = adaptors_pkg
    sys.modules["nestynet.adaptors.adaptors"] = adaptors_mod
    sys.modules["nestynet.adaptors.stacking_adaptors"] = stacking_mod


_install_nestynet_adaptor_stubs()

from nestynet_sr.sr_core.bridges import Add, U, Var
from nestynet_sr.sr_de.de_search import DESearchConfig, _maybe_expr_ir_de_rows
from nestynet_sr.sr_expr_ir.config import ExpressionIRConfig, expr_ir_active
from nestynet_sr.sr_expr_ir.core_bridge import canonical_key_core_ast
from nestynet_sr.sr_expr_ir.stats import ExpressionIRStats
from nestynet_sr.sr_expr_ir.tuple_bridge import canonical_key_tuple_ast, maybe_canonicalize_tuple_ast
from nestynet_sr.sr_search.config import FactorizedSearchConfig
from nestynet_sr.sr_search.factorized_search.expr_ast import build_pool
from nestynet_sr.sr_search.factorized_search.adapters.nestynet.stageb_runner import _expr_ir_kwargs_from_rule
from nestynet_sr.sr_search.stageB.rule_factorized_search import RuleFactorizedSearchFallback


def _qdag_cfg():
    return ExpressionIRConfig(expr_ir="qdag", canonicalize="safe", domain_mode="strict")


def test_expr_ir_default_is_inactive_for_tuple_bridge():
    node = ("add", ("var", 0), ("var", 1))
    assert not expr_ir_active(ExpressionIRConfig())
    assert maybe_canonicalize_tuple_ast(node, ExpressionIRConfig()) == node


def test_tuple_qdag_canonicalizes_ac_keys():
    cfg = _qdag_cfg()
    left = ("add", ("var", 1), ("var", 0))
    right = ("add", ("var", 0), ("var", 1))
    assert canonical_key_tuple_ast(left, cfg) == canonical_key_tuple_ast(right, cfg)


def test_tuple_build_pool_accepts_qdag_config_without_expanding_defaults():
    legacy = build_pool(2)
    stats = ExpressionIRStats()
    qdag_pool = build_pool(2, ir_cfg=_qdag_cfg(), ir_stats=stats)
    assert qdag_pool
    assert len(qdag_pool) <= len(legacy)
    assert stats.canonicalized_candidates > 0


def test_core_qdag_canonicalizes_de_rows_and_merges_sources():
    cfg = DESearchConfig(expr_ir="qdag", expr_canonicalize="safe")
    rows = [
        (Add(Var(0), U()), "baseline", "manual"),
        (Add(U(), Var(0)), "gs:test", "manual_gs"),
    ]
    out = _maybe_expr_ir_de_rows(rows, cfg, order=1)
    assert len(out) == 1
    term, source, family = out[0]
    assert canonical_key_core_ast(term, cfg) == canonical_key_core_ast(Add(Var(0), U()), cfg)
    assert "baseline" in source and "gs:test" in source
    assert "manual" in family and "manual_gs" in family
    assert getattr(cfg, "_expr_ir_last_de_report")["stats"]["canonicalized_candidates"] >= 1


def test_stageb_rule_forwards_expr_ir_kwargs():
    hp = FactorizedSearchConfig()
    hp.expr_ir = "qdag"
    hp.expr_canonicalize = "safe"
    hp.expr_deep_enable = True
    hp.expr_deep_max_depth = 7
    rule = RuleFactorizedSearchFallback(factorized_search_hp=hp)
    kwargs = _expr_ir_kwargs_from_rule(rule)
    assert kwargs["expr_ir"] == "qdag"
    assert kwargs["expr_canonicalize"] == "safe"
    assert kwargs["expr_deep_enable"] is True
    assert kwargs["expr_deep_max_depth"] == 7
