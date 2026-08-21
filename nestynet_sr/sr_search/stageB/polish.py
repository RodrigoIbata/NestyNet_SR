# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""Accepted-step polishing for Stage B.

This module is deliberately narrow.  It is only meant to run after Stage B has
already accepted a rewrite.  Whole-expression polish runs only once the live
expression is fully analytic; subtree shadow polish can also inspect newly
analytic subtrees while NN atoms remain.  Candidate forms are scored through the
normal Stage B fitting and acceptance policy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional

import torch

from nestynet_sr.sr_core.bridges import (
    AcosNode,
    AbsNode,
    AddNode,
    AsinNode,
    ArgNode,
    AtanNode,
    AtomNode,
    ConjNode,
    ConstNode,
    CosNode,
    ExpNode,
    ImagNode,
    LogNode,
    MulNode,
    Node,
    PowNode,
    RealNode,
    SinNode,
    ast_to_human_readable,
    clone_ast,
    collect_nn_atoms,
)
from nestynet_sr.sr_core.sympy_bridge import (
    coefficient_symbol_nodes_from_ast,
    sympy_to_nestynet,
)
from nestynet_sr.sr_search.model_selection import complexity_key as _complexity_key
from nestynet_sr.sr_search.polish_utils import (
    canonicalize_trig_phases,
    constant_code_cost,
    final_polish_snap_targets,
    numeric_constant_snap_candidates,
    rational_snap_targets,
    rationalize_float_exponents,
    snap_numeric_constants,
    sympy_expr_key,
)
from nestynet_sr.sr_search.representation import pretty_print_state

_MAX_STAGEB_POLISH_EXPR_CHARS = 50000


@dataclass
class StageBPolishConfig:
    """Configuration for accepted-step Stage B polishing."""

    enabled: bool = True
    commit: bool = True
    max_candidates: int = 32
    max_subtrees: int = 8
    use_subprocess: bool = True
    max_seconds: float = 300.0
    mem_fraction: float = 0.20


@dataclass
class StageBPolishResult:
    """Audit record for a Stage B polish attempt."""

    label: str
    expr_before: str
    expr_after: str
    reason: str
    base_loss: float
    cand_loss: float
    accepted: bool
    warnings: list[str] = field(default_factory=list)


@dataclass
class StageBSubtreePolishResult:
    """Shadow audit record for an analytic subtree polish attempt."""

    path: str
    label: str
    expr_before: str
    expr_after: str
    reason: str
    base_complexity: tuple[float, int]
    cand_complexity: tuple[float, int]
    accepted: bool
    subtree: Node
    cand_root: Optional[Node] = None
    full_root: Optional[Node] = None
    full_val_loss: Optional[float] = None
    full_policy_ok: Optional[bool] = None
    full_policy_reason: Optional[str] = None
    full_precheck_ok: Optional[bool] = None
    full_cand: Optional[Any] = None
    full_state: Optional[Any] = None


def _loader_is_multi(loader: Any) -> bool:
    return isinstance(loader, (list, tuple)) and bool(loader) and not isinstance(loader[0], torch.Tensor)


def _infer_nvars_from_loader(loader: Any) -> Optional[int]:
    if isinstance(loader, (list, tuple)) and loader and not isinstance(loader[0], torch.Tensor):
        loader = loader[0]
    try:
        batch = next(iter(loader))
    except Exception:
        return None
    try:
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        return int(x.shape[1])
    except Exception:
        return None


def _sympy_local_dict(nvars: int) -> dict[str, Any]:
    import sympy as sp

    # Some SR atom labels collide with SymPy helpers.  In particular
    # ``poly(x)`` would otherwise resolve to SymPy's Poly constructor, which is
    # not an Expr and triggers deprecation warnings when it appears inside Mul.
    sr_atom_functions = (
        "scale",
        "poly",
        "rpoly",
        "ratpoly",
        "rratpoly",
        "polylog",
        "rpolylog",
        "logshifted",
        "log_poly",
        "log_ratpoly",
        "exp_poly",
        "rexp_poly",
        "exp_rat",
        "ratio_poly",
        "inv_monomial",
        "rinv_monomial",
        "monomial",
        "sin_linear",
        "cos_linear",
        "tanh_leaf",
        "free_const",
    )
    loc: dict[str, Any] = {
        "sqrt": sp.sqrt,
        "sin": sp.sin,
        "cos": sp.cos,
        "tan": sp.tan,
        "tanh": sp.tanh,
        "asin": sp.asin,
        "acos": sp.acos,
        "atan": sp.atan,
        "arcsin": sp.asin,
        "arccos": sp.acos,
        "arctan": sp.atan,
        "exp": sp.exp,
        "log": sp.log,
        "Abs": sp.Abs,
        "abs": sp.Abs,
        "pi": sp.pi,
        "E": sp.E,
    }
    loc.update({name: sp.Function(name) for name in sr_atom_functions})
    loc.update({f"x{i}": sp.Symbol(f"x{i}", real=True) for i in range(int(nvars))})
    return loc


def _parse_expression(expr_str: str, nvars: int):
    import sympy as sp

    return sp.sympify(
        str(expr_str).replace("^", "**"),
        locals=_sympy_local_dict(nvars),
        evaluate=False,
    )


def _iter_child_nodes(node: Node) -> list[tuple[str, Node]]:
    if isinstance(node, (AddNode, MulNode)):
        return [("left", node.left), ("right", node.right)]
    if isinstance(node, PowNode):
        return [("base", node.base)]
    if isinstance(node, (LogNode, ExpNode, SinNode, CosNode, AsinNode, AcosNode, AtanNode, ConjNode, RealNode, ImagNode, AbsNode, ArgNode)):
        return [("arg", node.arg)]
    return []


def _walk_ast(node: Node, path: tuple[str, ...] = ()) -> list[tuple[tuple[str, ...], Node]]:
    out = [(path, node)]
    for name, child in _iter_child_nodes(node):
        out.extend(_walk_ast(child, path + (name,)))
    return out


def _path_to_str(path: tuple[str, ...]) -> str:
    return "root" if not path else "root." + ".".join(path)


def _path_from_str(path: str) -> tuple[str, ...]:
    parts = [p for p in str(path).split(".") if p]
    if not parts:
        return ()
    if parts[0] != "root":
        raise ValueError(f"subtree path must start with 'root', got {path!r}")
    return tuple(parts[1:])


def replace_subtree_at_path(root: Node, path: str, replacement: Node) -> Node:
    """Return a cloned AST with one subtree replaced by ``replacement``."""

    segments = _path_from_str(path)

    def rec(node: Node, rest: tuple[str, ...]) -> Node:
        if not rest:
            return clone_ast(replacement)
        head, tail = rest[0], rest[1:]
        if isinstance(node, AddNode):
            if head == "left":
                return AddNode(rec(node.left, tail), clone_ast(node.right))
            if head == "right":
                return AddNode(clone_ast(node.left), rec(node.right, tail))
        if isinstance(node, MulNode):
            if head == "left":
                return MulNode(rec(node.left, tail), clone_ast(node.right))
            if head == "right":
                return MulNode(clone_ast(node.left), rec(node.right, tail))
        if isinstance(node, PowNode):
            if head == "base":
                return PowNode(base=rec(node.base, tail), exponent=float(node.exponent))
        if isinstance(node, (LogNode, ExpNode, SinNode, CosNode, AsinNode, AcosNode, AtanNode, ConjNode, RealNode, ImagNode, AbsNode, ArgNode)):
            if head == "arg":
                return type(node)(arg=rec(node.arg, tail))
        raise ValueError(f"invalid subtree path segment {head!r} for {type(node).__name__}")

    return rec(root, segments)


def _node_fingerprint(node: Node) -> str:
    try:
        return repr(node)
    except Exception:
        return f"{type(node).__name__}:{id(node)}"


def _node_size(node: Node) -> int:
    return 1 + sum(_node_size(child) for _, child in _iter_child_nodes(node))


def _is_trivial_analytic_subtree(node: Node) -> bool:
    if isinstance(node, ConstNode):
        return True
    if isinstance(node, AtomNode):
        kind = str(getattr(node, "kind", "")).lower()
        return kind in ("var", "x", "input")
    return False


def _is_path_under(path: tuple[str, ...], ancestor: tuple[str, ...]) -> bool:
    return len(path) > len(ancestor) and path[: len(ancestor)] == ancestor


def _local_sympy_complexity(expr: Any) -> tuple[float, int]:
    import sympy as sp

    try:
        ops = float(sp.count_ops(expr, visual=False))
    except Exception:
        ops = float("inf")
    try:
        const_cost, n_long = constant_code_cost(
            expr,
            snap_targets=final_polish_snap_targets(),
            snap_rel_tol=5.0e-4,
        )
    except Exception:
        const_cost, n_long = 0.0, 0
    return (float(ops) + float(const_cost) + 4.0 * float(n_long), int(len(str(expr))))


def _unit_status_for_subtree(ctx: Any, subtree: Node) -> tuple[bool, str]:
    if not bool(getattr(ctx, "enforce_units", False)):
        return True, "units-unchecked"
    spec = getattr(ctx, "units_spec", None)
    if spec is None:
        return True, "units-unchecked"
    try:
        from nestynet_sr.sr_core.units import compute_node_domains
    except Exception as exc:
        return False, f"units-domain-unavailable({exc})"

    try:
        domains = compute_node_domains(ctx.state.root, spec)
    except Exception as exc:
        return False, f"units-domain-error({exc})"
    if domains is None:
        return False, "units-domain-inconsistent"
    dom = domains.get(id(subtree))
    if dom is None:
        return False, "units-domain-missing"
    try:
        if bool(dom.is_pinned()):
            return True, "units-pinned"
        return False, f"units-domain-unpinned(rank={dom.rank()})"
    except Exception:
        return False, "units-domain-unknown"


def select_new_analytic_subtrees(
    current_root: Node,
    previous_root: Optional[Node],
    *,
    max_subtrees: int = 8,
) -> list[tuple[str, Node]]:
    """Select maximal changed analytic subtrees in the current AST."""
    old_by_path = {
        path: _node_fingerprint(node)
        for path, node in (_walk_ast(previous_root) if previous_root is not None else [])
    }
    selected: list[tuple[tuple[str, ...], Node]] = []
    for path, node in _walk_ast(current_root):
        if len(selected) >= int(max_subtrees):
            break
        if _is_trivial_analytic_subtree(node):
            continue
        try:
            if len(collect_nn_atoms(node)) > 0:
                continue
        except Exception:
            continue
        if _node_size(node) < 3:
            continue
        old_fp = old_by_path.get(path)
        if old_fp is not None and old_fp == _node_fingerprint(node):
            continue
        if any(_is_path_under(path, parent_path) for parent_path, _ in selected):
            continue
        selected.append((path, node))
    return [(_path_to_str(path), node) for path, node in selected]


def _drop_numeric_additive_terms(expr: Any) -> Any:
    import sympy as sp

    if not isinstance(expr, sp.Add):
        return expr
    kept = [arg for arg in expr.args if not getattr(arg, "is_number", False)]
    if len(kept) == len(expr.args):
        return expr
    if not kept:
        return sp.Integer(0)
    return sp.Add(*kept)


def _snap_numeric_candidates(expr: Any, *, per_number: int = 4) -> list[tuple[str, Any]]:
    import sympy as sp

    targets = rational_snap_targets(max_denominator=6)
    nums = [
        n
        for n in sorted(expr.atoms(sp.Number), key=lambda v: (str(type(v)), str(v)))
        if not isinstance(n, sp.Integer)
    ]
    if not nums:
        return []

    def _dist(num: Any, target: Any) -> float:
        try:
            return abs(float(num) - float(target))
        except Exception:
            return float("inf")

    out: list[tuple[str, Any]] = []

    nearest_map = {}
    for num in nums:
        ranked = sorted(targets, key=lambda t: (_dist(num, t), len(str(t)), str(t)))
        for target in ranked:
            if _dist(num, target) > 0.0:
                nearest_map[num] = target
                break
    if nearest_map:
        out.append(("snap_all_numeric_nearest", expr.xreplace(nearest_map)))

    for num in nums:
        ranked = sorted(targets, key=lambda t: (_dist(num, t), len(str(t)), str(t)))
        kept = 0
        for target in ranked:
            if _dist(num, target) <= 0.0:
                continue
            out.append((f"snap_numeric:{num}->{target}", expr.xreplace({num: target})))
            kept += 1
            if kept >= int(per_number):
                break

    return out


def _drop_additive_term_candidates(expr: Any) -> list[tuple[str, Any]]:
    import sympy as sp

    out: list[tuple[str, Any]] = []
    for sub in sp.preorder_traversal(expr):
        if not isinstance(sub, sp.Add) or len(sub.args) <= 1:
            continue
        for arg in sub.args:
            kept = [a for a in sub.args if a is not arg]
            if not kept:
                replacement = sp.Integer(0)
            else:
                replacement = sp.Add(*kept, evaluate=False)
            out.append((f"drop_additive_term:{arg}", expr.xreplace({sub: replacement})))
    return out


def _tie_opposite_coefficient_candidates(expr: Any, *, max_pairs: int = 8) -> list[tuple[str, Any]]:
    import sympy as sp

    # Only tie numeric coefficients that are actually coefficients of additive
    # terms.  Looking at all Number atoms is too broad: it also sees exponents
    # such as the ``1/2`` in sqrt(...), and replacing those can create enormous
    # irrational-power expressions that make SymPy simplification explode.
    pairs = []
    for sub in sp.preorder_traversal(expr):
        if not isinstance(sub, sp.Add) or len(sub.args) <= 1:
            continue
        terms = []
        for arg in sub.args:
            try:
                coeff, rest = arg.as_coeff_Mul()
            except Exception:
                continue
            if rest == 1 or isinstance(coeff, sp.Integer):
                continue
            terms.append((arg, coeff, rest))
        for i, (arg_a, coeff_a, rest_a) in enumerate(terms):
            for arg_b, coeff_b, rest_b in terms[i + 1:]:
                try:
                    af = float(coeff_a)
                    bf = float(coeff_b)
                except Exception:
                    continue
                if af == 0.0 or bf == 0.0 or af * bf >= 0.0:
                    continue
                pairs.append((abs(abs(af) - abs(bf)), sub, arg_a, coeff_a, rest_a, arg_b, coeff_b, rest_b, af, bf))

    out: list[tuple[str, Any]] = []
    for _, sub, arg_a, coeff_a, rest_a, arg_b, coeff_b, rest_b, af, bf in sorted(pairs, key=lambda row: row[0])[: int(max_pairs)]:
        mag = (abs(af) + abs(bf)) / 2.0
        try:
            mag_expr = sp.nsimplify(mag, rational=True)
        except Exception:
            mag_expr = sp.Float(mag, 16)
        new_a = (mag_expr if af > 0.0 else -mag_expr) * rest_a
        new_b = (mag_expr if bf > 0.0 else -mag_expr) * rest_b
        new_args = []
        for arg in sub.args:
            if arg == arg_a:
                new_args.append(new_a)
            elif arg == arg_b:
                new_args.append(new_b)
            else:
                new_args.append(arg)
        replacement = sp.Add(*new_args, evaluate=False)
        out.append((f"tie_opposite_coefficients:{coeff_a},{coeff_b}", expr.xreplace({sub: replacement})))
    return out


def _candidate_exprs(expr: Any, *, max_candidates: int) -> list[tuple[str, Any]]:
    import sympy as sp

    raw: list[tuple[str, Any]] = []

    def add(label: str, fn):
        try:
            raw.append((label, fn()))
        except Exception:
            pass

    add("rationalize_float_exponents", lambda: rationalize_float_exponents(expr))
    add("cancel", lambda: sp.cancel(expr))
    add("factor", lambda: sp.factor(expr))
    add("factor_terms", lambda: sp.factor_terms(expr))
    add("together", lambda: sp.together(expr))
    add("together_cancel", lambda: sp.cancel(sp.together(expr)))
    add("powsimp", lambda: sp.powsimp(expr, force=True))
    add("trigsimp", lambda: sp.trigsimp(expr))
    add("canonicalize_trig_phases", lambda: canonicalize_trig_phases(expr))
    add("simplify", lambda: sp.simplify(expr))
    add("drop_numeric_additive_terms", lambda: _drop_numeric_additive_terms(expr))
    add("nsimplify", lambda: sp.nsimplify(expr, rational=True))
    raw.extend(_snap_numeric_candidates(expr))
    add(
        "snap_symbolic_constants",
        lambda: snap_numeric_constants(
            expr,
            snap_targets=final_polish_snap_targets(),
            snap_rel_tol=5.0e-4,
        ),
    )
    raw.extend(
        numeric_constant_snap_candidates(
            expr,
            snap_targets=final_polish_snap_targets(),
            snap_rel_tol=5.0e-4,
            per_number=4,
        )
    )
    raw.extend(_drop_additive_term_candidates(expr))
    raw.extend(_tie_opposite_coefficient_candidates(expr))

    try:
        coeff, rest = expr.as_coeff_Mul()
        if coeff != 1 and rest != 1:
            raw.append(("drop_global_scale", rest))
            raw.append(("snap_global_scale_nsimplify", sp.nsimplify(coeff, rational=True) * rest))
    except Exception:
        pass

    out: list[tuple[str, Any]] = []
    seen: set[str] = set()
    try:
        base_key = sympy_expr_key(expr)
    except Exception:
        base_key = str(expr)
    for label, cand in raw:
        if cand is None:
            continue
        try:
            cand = sp.simplify(cand)
        except Exception:
            pass
        try:
            key = sympy_expr_key(cand)
        except Exception:
            key = str(cand)
        if key == base_key or key in seen:
            continue
        seen.add(key)
        out.append((label, cand))
        if len(out) >= int(max_candidates):
            break
    return out


def stageB_polish_expr_candidates_worker(
    *,
    expr_str: str,
    nvars: int,
    max_candidates: int,
) -> dict[str, Any]:
    """Generate Stage-B polish expression candidates from a string.

    This function is intentionally importable and JSON-friendly so it can run
    inside ``postprocess_guard.run_guarded_function``.  It contains the risky
    SymPy work: parsing, candidate simplification, expression-keying, and local
    complexity accounting.
    """

    expr = _parse_expression(expr_str, int(nvars))
    base_cx = _local_sympy_complexity(expr)
    records: list[dict[str, Any]] = []
    for label, expr_after in _candidate_exprs(
        expr,
        max_candidates=max(1, int(max_candidates)),
    ):
        cand_cx = _local_sympy_complexity(expr_after)
        expr_after_str = str(expr_after)
        if len(expr_after_str) > _MAX_STAGEB_POLISH_EXPR_CHARS:
            continue
        records.append(
            {
                "label": str(label),
                "expr": expr_after_str,
                "complexity": (float(cand_cx[0]), int(cand_cx[1])),
            }
        )
    return {
        "ok": True,
        "base_complexity": (float(base_cx[0]), int(base_cx[1])),
        "candidates": records,
    }


def _stageB_polish_candidate_records(
    ctx: Any,
    *,
    expr_str: str,
    nvars: int,
    config: StageBPolishConfig,
    label: str,
) -> dict[str, Any]:
    """Return guarded or in-process polish candidate records."""

    max_candidates = max(1, int(getattr(config, "max_candidates", 32) or 32))
    if not bool(getattr(config, "use_subprocess", True)):
        try:
            return stageB_polish_expr_candidates_worker(
                expr_str=expr_str,
                nvars=int(nvars),
                max_candidates=max_candidates,
            )
        except Exception as exc:
            return {
                "ok": False,
                "reason": f"in-process candidate generation failed ({exc})",
            }

    max_seconds = float(getattr(config, "max_seconds", 300.0) or 300.0)
    mem_fraction = float(getattr(config, "mem_fraction", 0.20) or 0.20)
    try:
        ctx.log(
            f"[Stage B polish] Running {label} candidate generation in guarded worker "
            f"(max_seconds={max_seconds:.1f}, mem_fraction={mem_fraction:.3g})."
        )
    except Exception:
        pass

    try:
        from nestynet_sr.sr_search.postprocess_guard import run_guarded_function

        outcome = run_guarded_function(
            "nestynet_sr.sr_search.stageB.polish:stageB_polish_expr_candidates_worker",
            kwargs={
                "expr_str": str(expr_str),
                "nvars": int(nvars),
                "max_candidates": max_candidates,
            },
            max_seconds=max_seconds,
            mem_fraction=mem_fraction,
            label=f"stageB_polish_{label}",
        )
    except Exception as exc:
        return {
            "ok": False,
            "reason": f"guarded candidate worker failed to launch ({exc})",
        }

    if bool(outcome.get("ok", False)):
        result = outcome.get("result")
        if isinstance(result, dict) and bool(result.get("ok", False)):
            result = dict(result)
            result["guard_meta"] = {
                "status": outcome.get("status"),
                "returncode": outcome.get("returncode"),
                "memory_limit_bytes": outcome.get("memory_limit_bytes"),
                "max_seconds": outcome.get("max_seconds"),
                "limit_meta": outcome.get("limit_meta"),
            }
            return result
        return {
            "ok": False,
            "reason": "guarded candidate worker returned malformed result",
            "guard_outcome": outcome,
        }

    reason = outcome.get("reason") or outcome.get("error") or outcome.get("status") or "failed"
    return {
        "ok": False,
        "reason": f"guarded candidate worker {reason}",
        "guard_outcome": outcome,
    }


def _candidate_rank_key(ctx: Any, state: Any, expr_str: str) -> tuple[float, int, float, int]:
    try:
        n_params = int(state.model.num_parameters())
    except Exception:
        n_params = 0
    try:
        count_weight = float(getattr(ctx.lm_hp, "select_count_weight", 1.0))
    except Exception:
        count_weight = 1.0
    try:
        cx = _complexity_key(state.root, n_params, count_weight=count_weight)
    except Exception:
        cx = (float("inf"), int(1e18))
    return (float(cx[0]), int(cx[1]), float(state.val_loss), int(len(expr_str)))


def _rescore_full_subtree_candidate(
    ctx: Any,
    *,
    path: str,
    local_label: str,
    local_expr_after: str,
    cand_root: Node,
) -> tuple[Optional[Node], Optional[float], bool, str, bool, Optional[Any], Optional[Any]]:
    """Splice a local subtree candidate into the full AST and run Stage B policy."""
    try:
        full_root = replace_subtree_at_path(ctx.state.root, path, cand_root)
    except Exception as exc:
        return None, None, False, f"full-splice-failed({exc})", False, None, None

    from .engine import Candidate

    cand = Candidate(
        label=f"stageB_polish_subtree:{local_label}",
        root=full_root,
        meta={
            "stageB_polish_subtree": True,
            "structural": False,
            "log": f"[Stage B polish]  Shadow trying subtree {path}: {local_expr_after}",
        },
    )
    try:
        pre = ctx.precheck_candidate("stageB_polish_subtree", cand, record_attempt=False)
    except Exception as exc:
        return full_root, None, False, f"full-precheck-error({exc})", False, cand, None
    if pre is not None and not bool(getattr(pre, "ok", False)):
        return full_root, None, False, f"full-precheck-reject({getattr(pre, 'reason', 'reject')})", False, cand, None

    try:
        cand_state = ctx.fit_candidate(cand)
    except Exception as exc:
        return full_root, None, False, f"full-fit-failed({exc})", True, cand, None

    cand_loss = float(getattr(cand_state, "val_loss", float("inf")))
    if not math.isfinite(cand_loss):
        return full_root, cand_loss, False, "full-fit-nonfinite-loss", True, cand, cand_state

    best_before = getattr(ctx, "best_val_loss", None)
    try:
        ok, reason = ctx.should_accept(cand, cand_state)
    except Exception as exc:
        ok, reason = False, f"full-policy-error({exc})"
    finally:
        if best_before is not None:
            try:
                ctx.best_val_loss = best_before
            except Exception:
                pass

    return full_root, cand_loss, bool(ok), str(reason or ("accepted" if ok else "rejected")), True, cand, cand_state


def shadow_polish_new_analytic_subtrees(
    ctx: Any,
    *,
    previous_root: Optional[Node],
    config: Optional[StageBPolishConfig] = None,
) -> list[StageBSubtreePolishResult]:
    """Shadow-polish newly analytic subtrees through full-expression policy.

    The live Stage B state is not mutated.  A local subtree simplification only
    becomes a shadow accept if the respliced full AST fits and passes the normal
    Stage B acceptance policy.
    """
    config = config or StageBPolishConfig()
    if not bool(config.enabled):
        return []
    nvars = _infer_nvars_from_loader(getattr(ctx, "val_loader", None))
    if nvars is None:
        nvars = _infer_nvars_from_loader(getattr(ctx, "train_loader", None))
    if nvars is None:
        return []

    max_subtrees = int(getattr(config, "max_subtrees", 8) or 8)
    out: list[StageBSubtreePolishResult] = []
    for path, subtree in select_new_analytic_subtrees(
        ctx.state.root,
        previous_root,
        max_subtrees=max_subtrees,
    ):
        unit_ok, unit_reason = _unit_status_for_subtree(ctx, subtree)
        expr_before = ast_to_human_readable(subtree)
        try:
            coefficient_nodes = coefficient_symbol_nodes_from_ast(subtree)
        except Exception:
            coefficient_nodes = {}
        if not unit_ok:
            out.append(
                StageBSubtreePolishResult(
                    path=path,
                    label="stageB_polish_subtree",
                    expr_before=expr_before,
                    expr_after=expr_before,
                    reason=unit_reason,
                    base_complexity=(float("inf"), int(len(expr_before))),
                    cand_complexity=(float("inf"), int(len(expr_before))),
                    accepted=False,
                    subtree=subtree,
                    cand_root=None,
                )
            )
            continue

        records = _stageB_polish_candidate_records(
            ctx,
            expr_str=expr_before,
            nvars=int(nvars),
            config=config,
            label=f"subtree:{path}",
        )
        if not bool(records.get("ok", False)):
            reason = str(records.get("reason", "candidate-generation-failed"))
            out.append(
                StageBSubtreePolishResult(
                    path=path,
                    label="stageB_polish_subtree",
                    expr_before=expr_before,
                    expr_after=expr_before,
                    reason=reason,
                    base_complexity=(float("inf"), int(len(expr_before))),
                    cand_complexity=(float("inf"), int(len(expr_before))),
                    accepted=False,
                    subtree=subtree,
                    cand_root=None,
                )
            )
            continue

        base_complexity = records.get("base_complexity", (float("inf"), int(len(expr_before))))
        try:
            base_cx = (float(base_complexity[0]), int(base_complexity[1]))
        except Exception:
            base_cx = (float("inf"), int(len(expr_before)))
        best = None
        for rec in records.get("candidates", []):
            if not isinstance(rec, dict):
                continue
            label = str(rec.get("label", "polish"))
            expr_after_str = str(rec.get("expr", ""))
            cand_complexity = rec.get("complexity", (float("inf"), int(len(expr_after_str))))
            try:
                cand_cx = (float(cand_complexity[0]), int(cand_complexity[1]))
            except Exception:
                cand_cx = (float("inf"), int(len(expr_after_str)))
            if cand_cx >= base_cx:
                continue
            try:
                expr_after = _parse_expression(expr_after_str, int(nvars))
                cand_root = sympy_to_nestynet(
                    expr_after,
                    int(nvars),
                    symbol_nodes=coefficient_nodes,
                )
            except Exception:
                cand_root = None
            key = (cand_cx[0], cand_cx[1], expr_after_str)
            if best is None or key < best[0]:
                best = (key, label, expr_after_str, cand_cx, cand_root)

        if best is None:
            out.append(
                StageBSubtreePolishResult(
                    path=path,
                    label="stageB_polish_subtree",
                    expr_before=str(expr_before),
                    expr_after=str(expr_before),
                    reason="no-local-simpler-polish",
                    base_complexity=base_cx,
                    cand_complexity=base_cx,
                    accepted=False,
                    subtree=subtree,
                    cand_root=None,
                )
            )
            continue

        _, label, expr_after, cand_cx, cand_root = best
        if cand_root is None:
            out.append(
                StageBSubtreePolishResult(
                    path=path,
                    label=f"stageB_polish_subtree:{label}",
                    expr_before=str(expr_before),
                    expr_after=str(expr_after),
                    reason="local-simpler-but-uncompiled",
                    base_complexity=base_cx,
                    cand_complexity=cand_cx,
                    accepted=False,
                    subtree=subtree,
                    cand_root=None,
                    full_policy_ok=False,
                    full_policy_reason="local-simpler-but-uncompiled",
                    full_precheck_ok=False,
                )
            )
            continue

        full_root, full_loss, full_ok, full_reason, full_precheck_ok, full_cand, full_state = _rescore_full_subtree_candidate(
            ctx,
            path=path,
            local_label=label,
            local_expr_after=str(expr_after),
            cand_root=cand_root,
        )
        out.append(
            StageBSubtreePolishResult(
                path=path,
                label=f"stageB_polish_subtree:{label}",
                expr_before=str(expr_before),
                expr_after=str(expr_after),
                reason=f"shadow-full-policy({unit_reason}; {full_reason})",
                base_complexity=base_cx,
                cand_complexity=cand_cx,
                accepted=bool(full_ok),
                subtree=subtree,
                cand_root=cand_root,
                full_root=full_root,
                full_val_loss=full_loss,
                full_policy_ok=bool(full_ok),
                full_policy_reason=full_reason,
                full_precheck_ok=bool(full_precheck_ok),
                full_cand=full_cand,
                full_state=full_state,
            )
        )
    return out


def build_fully_analytic_polish_candidate(
    ctx: Any,
    config: Optional[StageBPolishConfig] = None,
) -> Optional[tuple[Any, Any, StageBPolishResult, str]]:
    """Return the best acceptable fully analytic polish candidate, if any.

    The returned tuple is ``(candidate, candidate_state, result, accept_reason)``.
    No mutation is performed here.
    """

    config = config or StageBPolishConfig()
    if not bool(config.enabled):
        return None
    if _loader_is_multi(getattr(ctx, "train_loader", None)) or _loader_is_multi(getattr(ctx, "val_loader", None)):
        ctx.log("[Stage B polish] skipped: multi-dataset polishing is not enabled yet")
        return None

    state = getattr(ctx, "state", None)
    if state is None:
        return None
    try:
        if len(collect_nn_atoms(state.root)) > 0:
            return None
    except Exception:
        if getattr(state, "num_nn_atoms", None) not in (0, None):
            return None

    nvars = _infer_nvars_from_loader(getattr(ctx, "val_loader", None))
    if nvars is None:
        nvars = _infer_nvars_from_loader(getattr(ctx, "train_loader", None))
    if nvars is None:
        ctx.log("[Stage B polish] skipped: could not infer input dimensionality")
        return None

    try:
        expr_before = pretty_print_state(state, sig=16)
        coefficient_nodes = coefficient_symbol_nodes_from_ast(state.root)
    except Exception as exc:
        ctx.log(f"[Stage B polish] skipped: pretty-print failed ({exc})")
        return None

    from .engine import Candidate

    best = None
    records = _stageB_polish_candidate_records(
        ctx,
        expr_str=expr_before,
        nvars=int(nvars),
        config=config,
        label="fully_analytic",
    )
    if not bool(records.get("ok", False)):
        ctx.log(
            "[Stage B polish] skipped: "
            f"{records.get('reason', 'candidate generation failed')}"
        )
        return None

    for rec in records.get("candidates", []):
        if not isinstance(rec, dict):
            continue
        label = str(rec.get("label", "polish"))
        expr_after_str = str(rec.get("expr", ""))
        try:
            expr_after = _parse_expression(expr_after_str, int(nvars))
            cand_root = sympy_to_nestynet(
                expr_after,
                int(nvars),
                symbol_nodes=coefficient_nodes,
            )
        except Exception:
            continue
        cand = Candidate(
            label=f"stageB_polish:{label}",
            root=cand_root,
            meta={
                "stageB_polish": True,
                "structural": False,
                "log": f"[Stage B polish]  Trying {label}: {expr_after_str}",
            },
        )
        try:
            pre = ctx.precheck_candidate("stageB_polish", cand, record_attempt=False)
        except Exception:
            pre = None
        if pre is not None and not getattr(pre, "ok", False):
            continue
        try:
            cand_state = ctx.fit_candidate(cand)
        except Exception:
            # Fully-analytic polish is opportunistic.  Immediately after some
            # terminal rewrites, the visible AST can be valid while transient
            # model/leaf bookkeeping is not yet suitable for a shadow refit.
            # Skip that polish candidate and keep the already accepted state.
            continue
        cand_loss = float(getattr(cand_state, "val_loss", float("inf")))
        if not math.isfinite(cand_loss):
            continue
        try:
            ok, reason = ctx.should_accept(cand, cand_state)
        except Exception:
            continue
        result = StageBPolishResult(
            label=label,
            expr_before=str(expr_before),
            expr_after=str(expr_after_str),
            reason=str(reason or ("accepted" if ok else "rejected")),
            base_loss=float(getattr(state, "val_loss", float("inf"))),
            cand_loss=float(cand_loss),
            accepted=bool(ok),
        )
        if not ok:
            continue
        key = _candidate_rank_key(ctx, cand_state, expr_after_str)
        if best is None or key < best[0]:
            best = (key, cand, cand_state, result, reason or "accepted")

    if best is None:
        return None
    _, cand, cand_state, result, reason = best
    return cand, cand_state, result, str(reason)


__all__ = [
    "StageBPolishConfig",
    "StageBPolishResult",
    "StageBSubtreePolishResult",
    "build_fully_analytic_polish_candidate",
    "replace_subtree_at_path",
    "select_new_analytic_subtrees",
    "stageB_polish_expr_candidates_worker",
    "shadow_polish_new_analytic_subtrees",
]
