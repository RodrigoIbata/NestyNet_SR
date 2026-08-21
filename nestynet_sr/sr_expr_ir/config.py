# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""Configuration helpers for the opt-in quotient-DAG expression IR."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any, Mapping


_IR_MODES = {"ast", "qdag", "qdag-egraph"}
_CANON_MODES = {"off", "safe", "common-domain", "aggressive"}
_DOMAIN_MODES = {"strict", "common-domain", "sample-guarded"}
_SEED_MODES = {"none", "unit", "affine", "known-lie", "jet", "all"}


@dataclass
class ExpressionIRConfig:
    expr_ir: str = "ast"
    canonicalize: str = "off"
    domain_mode: str = "strict"

    qdag_hash_cons: bool = True
    qdag_flatten_ac: bool = True
    qdag_combine_like_terms: bool = True
    qdag_combine_powers: bool = True
    qdag_constant_fold: bool = True
    qdag_polynomial_islands: bool = False
    qdag_rational_islands: bool = False
    qdag_max_nodes: int = 200_000
    qdag_max_terms_per_add: int = 128
    qdag_max_factors_per_mul: int = 128

    symmetry_signatures: bool = False
    symmetry_prune: bool = False
    invariant_coordinates: bool = False
    invariant_seeds: str = "none"

    # Generic deep-search controls, separate from legacy max_depth.
    deep_enable: bool = False
    deep_max_depth: int | None = None
    qdag_max_cost: float | None = None
    qdag_max_unique: int | None = None
    max_lowered_depth: int | None = None
    max_lowered_size: int | None = None

    # FSS/GS surfaces. Defaults keep GS out of FSS proposal construction.
    gs_fss_score: bool = False
    gs_fss_aux_generator: bool = False
    gs_fss_max_aux_atoms: int = 0
    gs_fss_max_seed_blocks: int = 0
    gs_fss_max_source_fraction: float = 0.0

    egraph_enable: bool = False
    egraph_rules: str = "safe"
    egraph_max_input_size: int = 64
    egraph_max_eclasses: int = 5_000
    egraph_max_enodes: int = 20_000
    egraph_max_iters: int = 8
    egraph_time_ms: int = 50

    report: bool = False
    fallback_on_error: bool = True
    strict_errors: bool = False
    debug_dump_examples: int = 0


def normalize_expr_ir_mode(value: str | None) -> str:
    mode = str(value or "ast").strip().lower().replace("_", "-")
    return mode if mode in _IR_MODES else "ast"


def normalize_canonicalize_mode(value: str | None) -> str:
    mode = str(value or "off").strip().lower().replace("_", "-")
    return mode if mode in _CANON_MODES else "off"


def normalize_domain_mode(value: str | None) -> str:
    mode = str(value or "strict").strip().lower().replace("_", "-")
    return mode if mode in _DOMAIN_MODES else "strict"


def normalize_invariant_seeds(value: str | None) -> str:
    mode = str(value or "none").strip().lower().replace("_", "-")
    return mode if mode in _SEED_MODES else "none"


def _prefixed_name(name: str) -> str:
    if name == "expr_ir":
        return "expr_ir"
    if name == "canonicalize":
        return "expr_canonicalize"
    if name == "domain_mode":
        return "expr_domain_mode"
    return f"expr_{name}"


def _get(obj: object | None, name: str, default: Any) -> Any:
    prefixed = _prefixed_name(name)
    if obj is None:
        return default
    if isinstance(obj, Mapping):
        if name in obj:
            return obj[name]
        if prefixed in obj:
            return obj[prefixed]
        return default
    if hasattr(obj, name):
        return getattr(obj, name)
    if hasattr(obj, prefixed):
        return getattr(obj, prefixed)
    return default


def _bool(value: Any, default: bool) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, str):
        return value.strip().lower() not in {"", "0", "false", "no", "off"}
    return bool(value)


def _int(value: Any, default: int | None) -> int | None:
    if value is None:
        return default
    try:
        return int(value)
    except Exception:
        return default


def _float(value: Any, default: float | None) -> float | None:
    if value is None:
        return default
    try:
        return float(value)
    except Exception:
        return default


def coerce_expr_ir_config(obj: object | None) -> ExpressionIRConfig:
    if isinstance(obj, ExpressionIRConfig):
        cfg = obj
    else:
        base = ExpressionIRConfig()
        values = {f.name: _get(obj, f.name, getattr(base, f.name)) for f in fields(ExpressionIRConfig)}
        cfg = ExpressionIRConfig(**values)

    cfg.expr_ir = normalize_expr_ir_mode(cfg.expr_ir)
    cfg.canonicalize = normalize_canonicalize_mode(cfg.canonicalize)
    cfg.domain_mode = normalize_domain_mode(cfg.domain_mode)
    cfg.invariant_seeds = normalize_invariant_seeds(cfg.invariant_seeds)

    for name in (
        "qdag_hash_cons",
        "qdag_flatten_ac",
        "qdag_combine_like_terms",
        "qdag_combine_powers",
        "qdag_constant_fold",
        "qdag_polynomial_islands",
        "qdag_rational_islands",
        "symmetry_signatures",
        "symmetry_prune",
        "invariant_coordinates",
        "deep_enable",
        "gs_fss_score",
        "gs_fss_aux_generator",
        "egraph_enable",
        "report",
        "fallback_on_error",
        "strict_errors",
    ):
        setattr(cfg, name, _bool(getattr(cfg, name), bool(getattr(ExpressionIRConfig(), name))))

    cfg.qdag_max_nodes = max(1, int(_int(cfg.qdag_max_nodes, 200_000) or 200_000))
    cfg.qdag_max_terms_per_add = max(1, int(_int(cfg.qdag_max_terms_per_add, 128) or 128))
    cfg.qdag_max_factors_per_mul = max(1, int(_int(cfg.qdag_max_factors_per_mul, 128) or 128))
    cfg.deep_max_depth = _int(cfg.deep_max_depth, None)
    cfg.qdag_max_cost = _float(cfg.qdag_max_cost, None)
    cfg.qdag_max_unique = _int(cfg.qdag_max_unique, None)
    cfg.max_lowered_depth = _int(cfg.max_lowered_depth, None)
    cfg.max_lowered_size = _int(cfg.max_lowered_size, None)
    cfg.gs_fss_max_aux_atoms = max(0, int(_int(cfg.gs_fss_max_aux_atoms, 0) or 0))
    cfg.gs_fss_max_seed_blocks = max(0, int(_int(cfg.gs_fss_max_seed_blocks, 0) or 0))
    cfg.gs_fss_max_source_fraction = max(0.0, float(_float(cfg.gs_fss_max_source_fraction, 0.0) or 0.0))
    cfg.egraph_max_input_size = max(1, int(_int(cfg.egraph_max_input_size, 64) or 64))
    cfg.egraph_max_eclasses = max(1, int(_int(cfg.egraph_max_eclasses, 5_000) or 5_000))
    cfg.egraph_max_enodes = max(1, int(_int(cfg.egraph_max_enodes, 20_000) or 20_000))
    cfg.egraph_max_iters = max(1, int(_int(cfg.egraph_max_iters, 8) or 8))
    cfg.egraph_time_ms = max(1, int(_int(cfg.egraph_time_ms, 50) or 50))
    cfg.debug_dump_examples = max(0, int(_int(cfg.debug_dump_examples, 0) or 0))
    return cfg


def expr_ir_active(cfg: ExpressionIRConfig | object | None) -> bool:
    c = coerce_expr_ir_config(cfg)
    return bool(
        c.expr_ir != "ast"
        and (
            c.canonicalize != "off"
            or c.egraph_enable
            or c.symmetry_signatures
            or c.symmetry_prune
            or c.invariant_coordinates
            or c.deep_enable
            or c.gs_fss_score
            or c.gs_fss_aux_generator
        )
    )


def expression_ir_config_to_dict(cfg: ExpressionIRConfig | object | None) -> dict[str, Any]:
    return dict(asdict(coerce_expr_ir_config(cfg)))


def add_expr_ir_cli_args(parser: Any) -> Any:
    parser.add_argument("--expr-ir", choices=["ast", "qdag", "qdag-egraph"], default="ast")
    parser.add_argument("--expr-canonicalize", choices=["off", "safe", "common-domain", "aggressive"], default="off")
    parser.add_argument("--expr-domain-mode", choices=["strict", "common-domain", "sample-guarded"], default="strict")
    parser.add_argument("--expr-qdag-hash-cons", dest="expr_qdag_hash_cons", action="store_true", default=True)
    parser.add_argument("--no-expr-qdag-hash-cons", dest="expr_qdag_hash_cons", action="store_false")
    parser.add_argument("--expr-qdag-combine-like-terms", dest="expr_qdag_combine_like_terms", action="store_true", default=True)
    parser.add_argument("--no-expr-qdag-combine-like-terms", dest="expr_qdag_combine_like_terms", action="store_false")
    parser.add_argument("--expr-qdag-combine-powers", dest="expr_qdag_combine_powers", action="store_true", default=True)
    parser.add_argument("--no-expr-qdag-combine-powers", dest="expr_qdag_combine_powers", action="store_false")
    parser.add_argument("--expr-qdag-polynomial-islands", action="store_true", default=False)
    parser.add_argument("--expr-qdag-rational-islands", action="store_true", default=False)
    parser.add_argument("--expr-qdag-max-cost", type=float, default=None)
    parser.add_argument("--expr-qdag-max-unique", type=int, default=None)
    parser.add_argument("--expr-max-lowered-depth", type=int, default=None)
    parser.add_argument("--expr-max-lowered-size", type=int, default=None)
    parser.add_argument("--expr-deep-enable", action="store_true", default=False)
    parser.add_argument("--expr-deep-max-depth", type=int, default=None)
    parser.add_argument("--expr-symmetry-signatures", action="store_true", default=False)
    parser.add_argument("--expr-symmetry-prune", action="store_true", default=False)
    parser.add_argument("--expr-invariant-coordinates", action="store_true", default=False)
    parser.add_argument("--expr-invariant-seeds", choices=["none", "unit", "affine", "known-lie", "jet", "all"], default="none")
    parser.add_argument("--expr-gs-fss-score", action="store_true", default=False)
    parser.add_argument("--expr-gs-fss-aux-generator", action="store_true", default=False)
    parser.add_argument("--expr-gs-fss-max-aux-atoms", type=int, default=0)
    parser.add_argument("--expr-gs-fss-max-seed-blocks", type=int, default=0)
    parser.add_argument("--expr-gs-fss-max-source-fraction", type=float, default=0.0)
    parser.add_argument("--expr-egraph-enable", action="store_true", default=False)
    parser.add_argument("--expr-egraph-rules", type=str, default="safe")
    parser.add_argument("--expr-egraph-max-input-size", type=int, default=64)
    parser.add_argument("--expr-egraph-max-eclasses", type=int, default=5000)
    parser.add_argument("--expr-egraph-max-enodes", type=int, default=20000)
    parser.add_argument("--expr-egraph-max-iters", type=int, default=8)
    parser.add_argument("--expr-egraph-time-ms", type=int, default=50)
    parser.add_argument("--expr-report", action="store_true", default=False)
    parser.add_argument("--expr-fallback-on-error", dest="expr_fallback_on_error", action="store_true", default=True)
    parser.add_argument("--no-expr-fallback-on-error", dest="expr_fallback_on_error", action="store_false")
    parser.add_argument("--expr-strict-errors", action="store_true", default=False)
    parser.add_argument("--expr-debug-dump-examples", type=int, default=0)
    return parser


def expr_ir_arg_items(args: object) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for f in fields(ExpressionIRConfig):
        attr = _prefixed_name(f.name)
        if hasattr(args, attr):
            out[attr] = getattr(args, attr)
    return out


def apply_expr_ir_args_to_obj(args: object, target: object) -> object:
    for name, value in expr_ir_arg_items(args).items():
        setattr(target, name, value)
    return target


__all__ = [
    "ExpressionIRConfig",
    "add_expr_ir_cli_args",
    "apply_expr_ir_args_to_obj",
    "coerce_expr_ir_config",
    "expr_ir_active",
    "expr_ir_arg_items",
    "expression_ir_config_to_dict",
]
