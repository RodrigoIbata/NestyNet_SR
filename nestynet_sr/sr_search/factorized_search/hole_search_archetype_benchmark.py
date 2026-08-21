# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata

"""Tiny archetype smoke benchmark through the live hole_search market.

This suite is intentionally narrow. It exists to regression-test three concrete
subproblem archetypes through ``run_hole_search_action``:

- drifting constants / constant lift
- single-index coordinate lift
- near-miss subtree rescue via local edit routing

The benchmark keeps the live hole-search route construction, solver market,
preview scoring, and selected-route reporting. To keep the suite fast and
deterministic, it only pins the deepest backend needed for each archetype.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import pathlib
import random
import sys
from collections import defaultdict
from typing import Any, Iterator, Mapping, Sequence

import torch

ROOT = pathlib.Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import nestynet_sr.sr_search.factorized_search.constant_lift_solver as constant_lift_mod
import nestynet_sr.sr_search.factorized_search.coordinate_lift_solver as coordinate_lift_mod
import nestynet_sr.sr_search.factorized_search.hole_search as hole_search_mod
from nestynet_sr.sr_search.factorized_search.expr_ast import node_str
from nestynet_sr.sr_search.factorized_search.hole_search import HoleOpportunity, run_hole_search_action
from nestynet_sr.sr_search.factorized_search.subproblem_spec import (
    SubproblemSpec,
    WitnessBundle,
    wrap_subproblem_spec_payload,
)


REPO_ROOT = ROOT
DEFAULT_SUITE_MANIFEST = (
    REPO_ROOT
    / "examples"
    / "oracle_factorized_search"
    / "capability_suites"
    / "hole_search_archetypes_smoke.json"
)


def _jsonable(value: Any) -> Any:
    if torch.is_tensor(value):
        if value.ndim == 0:
            try:
                return float(value.item())
            except Exception:
                return None
        return value.detach().cpu().tolist()
    if isinstance(value, pathlib.Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in dict(value).items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, float):
        return float(value) if math.isfinite(value) else None
    try:
        scalar = float(value)
    except Exception:
        return str(value)
    return float(scalar) if math.isfinite(scalar) else None


def _write_json(payload: Mapping[str, Any], path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(dict(payload)), indent=2), encoding="utf-8")


def _load_json(path: pathlib.Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_hole_search_archetype_suite(
    path: str | pathlib.Path | None = None,
) -> tuple[pathlib.Path, dict[str, Any]]:
    manifest_path = pathlib.Path(path) if path is not None else DEFAULT_SUITE_MANIFEST
    payload = _load_json(manifest_path)
    cases = list(payload.get("cases") or [])
    if not cases:
        raise ValueError(f"No cases declared in hole-search archetype suite: {manifest_path}")
    return manifest_path, payload


def _safe_float(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def _coerce_route_names(
    raw: Sequence[Any] | Any | None,
    *,
    default: Sequence[str],
) -> tuple[str, ...]:
    values = raw if isinstance(raw, (list, tuple)) else default
    out: list[str] = []
    seen: set[str] = set()
    for item in list(values or ()):
        token = str(item or "").strip()
        if not token or token in seen:
            continue
        seen.add(token)
        out.append(token)
    return tuple(out or tuple(default))


def _make_local_problem_opportunity(
    *,
    spec: SubproblemSpec,
    parent_key: str,
    parent_expr,
    direction: str,
    branch_id: str,
    path_gain: float = 0.5,
    confidence: float = 0.9,
    valid_frac: float = 0.95,
) -> HoleOpportunity:
    return HoleOpportunity(
        parent_key=str(parent_key),
        parent_expr_str=str(node_str(parent_expr)),
        path=tuple(int(v) for v in tuple(spec.path or ())),
        target_mode=str(spec.target_mode or "identity"),
        beam_rank=0,
        spec_kind="local_problem",
        direction=str(direction or spec.direction or "inside_out"),
        branch_id=str(branch_id),
        path_gain=float(path_gain),
        confidence=float(confidence),
        valid_frac=float(valid_frac),
        target_mapping_kind=str(spec.target_mapping_kind or "affine"),
        spec_payload=wrap_subproblem_spec_payload(spec),
    )


def _build_drifting_constant_case(case: Mapping[str, Any], defaults: Mapping[str, Any]) -> dict[str, Any]:
    x_fit = torch.tensor(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [0.0, 2.0],
        ],
        dtype=torch.float64,
    )
    x_probe = torch.tensor(
        [
            [0.0, 0.5],
            [0.0, 1.5],
            [0.0, 2.5],
        ],
        dtype=torch.float64,
    )
    t_fit = x_fit[:, 1:2].clone()
    t_probe = x_probe[:, 1:2].clone()
    parent_node = ("add", ("var", 0), ("const", 0.0))
    spec = SubproblemSpec(
        problem_id=str(case.get("case_id", "drifting_constant") or "drifting_constant"),
        problem_kind="local_problem",
        parent_expr=parent_node,
        path=(2,),
        direction="inside_out",
        target_mode="identity",
        target_mapping_kind="affine",
        target_dim=None,
        continuation_frames=(),
        wrappers_left=0,
        recursion_level=1,
        active_vars=(1,),
        witness=WitnessBundle(
            x_fit=x_fit,
            t_fit=t_fit,
            x_probe=x_probe,
            t_probe=t_probe,
            masks={},
            diagnostics={
                "confidence": 0.9,
                "valid_frac": 0.95,
                "trace": ("constant_lift",),
                "dataset_ids": ["d0", "d1", "d2"],
                "dataset_metadata": {
                    "d0": {"temperature": 0.0},
                    "d1": {"temperature": 1.0},
                    "d2": {"temperature": 2.0},
                },
                "local_constants_by_experiment": {
                    "d0": {"local_leaf": 1.0, "stable_leaf": 5.0},
                    "d1": {"local_leaf": 2.0, "stable_leaf": 5.05},
                    "d2": {"local_leaf": 4.0, "stable_leaf": 4.95},
                },
            },
        ),
        metadata={"hole_sub": ("const", 0.0)},
    )
    opportunity = _make_local_problem_opportunity(
        spec=spec,
        parent_key=str(case.get("parent_key", spec.problem_id) or spec.problem_id),
        parent_expr=parent_node,
        direction="inside_out",
        branch_id=str(case.get("branch_id", "archetype_constant_lift") or "archetype_constant_lift"),
    )
    return {
        "case_id": str(case.get("case_id", spec.problem_id) or spec.problem_id),
        "archetype": "drifting_constant",
        "expected_route_names": _coerce_route_names(
            case.get("expected_route_names", None),
            default=("constant_lift_route",),
        ),
        "selected_preview_probe_mse_max": float(
            case.get(
                "selected_preview_probe_mse_max",
                defaults.get("selected_preview_probe_mse_max", 1.0e-6),
            )
            or 1.0e-6
        ),
        "parent_node": parent_node,
        "opportunity": opportunity,
        "x_fit": x_fit,
        "t_fit": t_fit,
        "x_probe": x_probe,
        "t_probe": t_probe,
        "nvars": 2,
        "backend_override_kind": "constant_lift",
    }


def _build_single_index_coordinate_case(case: Mapping[str, Any], defaults: Mapping[str, Any]) -> dict[str, Any]:
    x0_fit = torch.linspace(-1.0, 1.0, 17, dtype=torch.float64)
    x1_fit = torch.linspace(1.0, -1.0, 17, dtype=torch.float64)
    x_fit = torch.stack([x0_fit, x1_fit + 0.5 * x0_fit], dim=1)
    x0_probe = torch.linspace(-1.25, 1.25, 19, dtype=torch.float64)
    x1_probe = torch.linspace(1.25, -1.25, 19, dtype=torch.float64)
    x_probe = torch.stack([x0_probe, x1_probe + 0.5 * x0_probe], dim=1)
    z_fit = x_fit[:, 0:1] + x_fit[:, 1:2]
    z_probe = x_probe[:, 0:1] + x_probe[:, 1:2]
    t_fit = torch.sin(z_fit)
    t_probe = torch.sin(z_probe)
    grad_fit = torch.cos(z_fit).repeat(1, 2)
    grad_probe = torch.cos(z_probe).repeat(1, 2)
    parent_node = ("add", ("const", 1.0), ("var", 0))
    spec = SubproblemSpec(
        problem_id=str(case.get("case_id", "single_index_coordinate") or "single_index_coordinate"),
        problem_kind="local_problem",
        parent_expr=parent_node,
        path=(1,),
        direction="inside_out",
        target_mode="identity",
        target_mapping_kind="affine",
        target_dim=None,
        continuation_frames=(),
        wrappers_left=0,
        recursion_level=1,
        active_vars=(0, 1),
        witness=WitnessBundle(
            x_fit=x_fit,
            t_fit=t_fit,
            x_probe=x_probe,
            t_probe=t_probe,
            grad_fit=grad_fit,
            grad_probe=grad_probe,
            masks={},
            diagnostics={"confidence": 0.9, "valid_frac": 0.95, "trace": ("coordinate_lift",)},
        ),
        metadata={"hole_sub": ("var", 0)},
    )
    opportunity = _make_local_problem_opportunity(
        spec=spec,
        parent_key=str(case.get("parent_key", spec.problem_id) or spec.problem_id),
        parent_expr=parent_node,
        direction="inside_out",
        branch_id=str(case.get("branch_id", "archetype_coordinate_lift") or "archetype_coordinate_lift"),
    )
    return {
        "case_id": str(case.get("case_id", spec.problem_id) or spec.problem_id),
        "archetype": "single_index_coordinate",
        "expected_route_names": _coerce_route_names(
            case.get("expected_route_names", None),
            default=("coordinate_lift",),
        ),
        "selected_preview_probe_mse_max": float(
            case.get(
                "selected_preview_probe_mse_max",
                defaults.get("selected_preview_probe_mse_max", 1.0e-5),
            )
            or 1.0e-5
        ),
        "parent_node": parent_node,
        "opportunity": opportunity,
        "x_fit": x_fit,
        "t_fit": t_fit,
        "x_probe": x_probe,
        "t_probe": t_probe,
        "nvars": 2,
        "backend_override_kind": "coordinate_lift",
        "coordinate_z_fit": z_fit,
        "coordinate_z_probe": z_probe,
    }


def _build_tangent_edit_case(case: Mapping[str, Any], defaults: Mapping[str, Any]) -> dict[str, Any]:
    x_fit = torch.linspace(-1.0, 1.0, 17, dtype=torch.float64).unsqueeze(-1)
    x_probe = torch.linspace(-1.5, 1.5, 25, dtype=torch.float64).unsqueeze(-1)
    t_fit = torch.sin(x_fit)
    t_probe = torch.sin(x_probe)
    parent_node = ("var", 0)
    spec = SubproblemSpec(
        problem_id=str(case.get("case_id", "near_miss_tangent_edit") or "near_miss_tangent_edit"),
        problem_kind="local_problem",
        parent_expr=parent_node,
        path=(),
        direction="inside_out",
        target_mode="identity",
        target_mapping_kind="affine",
        target_dim=None,
        continuation_frames=(),
        wrappers_left=0,
        recursion_level=1,
        active_vars=(0,),
        witness=WitnessBundle(
            x_fit=x_fit,
            t_fit=t_fit,
            x_probe=x_probe,
            t_probe=t_probe,
            grad_fit=torch.cos(x_fit),
            grad_probe=torch.cos(x_probe),
            diagnostics={"confidence": 0.9, "valid_frac": 0.95, "trace": ("tangent_edit",)},
            masks={},
        ),
        metadata={"hole_sub": ("var", 0)},
    )
    opportunity = _make_local_problem_opportunity(
        spec=spec,
        parent_key=str(case.get("parent_key", spec.problem_id) or spec.problem_id),
        parent_expr=parent_node,
        direction="inside_out",
        branch_id=str(case.get("branch_id", "archetype_tangent_edit") or "archetype_tangent_edit"),
    )
    return {
        "case_id": str(case.get("case_id", spec.problem_id) or spec.problem_id),
        "archetype": "near_miss_tangent_edit",
        "expected_route_names": _coerce_route_names(
            case.get("expected_route_names", None),
            default=("tangent_edit", "soft_edit_search"),
        ),
        "selected_preview_probe_mse_max": float(
            case.get(
                "selected_preview_probe_mse_max",
                defaults.get("selected_preview_probe_mse_max", 1.0e-8),
            )
            or 1.0e-8
        ),
        "parent_node": parent_node,
        "opportunity": opportunity,
        "x_fit": x_fit,
        "t_fit": t_fit,
        "x_probe": x_probe,
        "t_probe": t_probe,
        "nvars": 1,
        "backend_override_kind": "tangent_edit",
    }


_CASE_BUILDERS = {
    "drifting_constant": _build_drifting_constant_case,
    "single_index_coordinate": _build_single_index_coordinate_case,
    "near_miss_tangent_edit": _build_tangent_edit_case,
}


@contextlib.contextmanager
def _patched_attr(obj: Any, name: str, value: Any) -> Iterator[None]:
    original = getattr(obj, name)
    setattr(obj, name, value)
    try:
        yield
    finally:
        setattr(obj, name, original)


@contextlib.contextmanager
def _case_backend_overrides(case_state: Mapping[str, Any]) -> Iterator[None]:
    kind = str(case_state.get("backend_override_kind", "") or "")
    with contextlib.ExitStack() as stack:
        if kind == "constant_lift":
            def _fake_solve_constant_lift_task(**kwargs):
                return {
                    "solver": "factorized_search",
                    "expr": "x0",
                    "expr_ast": ["var", 0],
                    "mapping": {"kind": "poly", "coeffs": [0.0, 1.0], "mu": 0.0, "std": 1.0},
                    "fit_mse": 1.0e-6,
                    "probe_mse": 1.0e-6,
                    "baseline_mse": 1.0,
                    "improvement_ratio": 100.0,
                    "regime_ids": ["d0", "d1", "d2"],
                    "feature_names": ["temperature"],
                    "feature_source": "dataset_metadata",
                }

            stack.enter_context(
                _patched_attr(constant_lift_mod, "solve_constant_lift_task", _fake_solve_constant_lift_task)
            )
        elif kind == "coordinate_lift":
            z_fit = case_state["coordinate_z_fit"]
            z_probe = case_state["coordinate_z_probe"]

            def _fake_run_explorer(**kwargs):
                x_fit_data = kwargs.get("x_fit_data", None)
                x_probe_data = kwargs.get("x_probe_data", None)
                if torch.is_tensor(x_fit_data) and torch.is_tensor(x_probe_data):
                    if torch.allclose(x_fit_data, z_fit) and torch.allclose(x_probe_data, z_probe):
                        return [
                            {
                                "toy_ast": ("sin", ("var", 0)),
                                "mapping": {"kind": "poly", "coeffs": [0.0, 1.0], "mu": 0.0, "std": 1.0},
                                "mse_raw": 1.0e-4,
                                "mse_eff": 1.0e-4,
                            }
                        ]
                return []

            stack.enter_context(_patched_attr(coordinate_lift_mod, "run_explorer", _fake_run_explorer))
        elif kind == "tangent_edit":
            def _disable_inverse_followup(**kwargs):
                return {"rows": [], "solver_meta": {"status": "disabled_for_archetype"}}

            stack.enter_context(_patched_attr(coordinate_lift_mod, "run_explorer", lambda **kwargs: []))
            stack.enter_context(
                _patched_attr(hole_search_mod, "solve_local_problem_spec_preview_rows", _disable_inverse_followup)
            )
        yield


def _run_case(
    case_state: Mapping[str, Any],
    *,
    defaults: Mapping[str, Any],
    save_dir: pathlib.Path | None,
) -> dict[str, Any]:
    hole_defaults = dict(defaults.get("hole_search", {}) or {})
    selected_preview_probe_mse_max = _safe_float(case_state.get("selected_preview_probe_mse_max", None))
    with _case_backend_overrides(case_state):
        expr, meta = run_hole_search_action(
            case_state["opportunity"],
            parent_node=case_state["parent_node"],
            parent_mapping={"kind": "poly", "coeffs": [0.0, 1.0], "mu": 0.0, "std": 1.0},
            x_fit=case_state["x_fit"],
            y_fit=case_state["t_fit"].squeeze(-1),
            x_probe=case_state["x_probe"],
            y_probe=case_state["t_probe"].squeeze(-1),
            pool_nodes=[],
            pool_phi_fit=case_state["x_fit"][:, :0],
            pool_phi_probe=case_state["x_probe"][:, :0],
            pool_dims=[],
            rng=random.Random(int(hole_defaults.get("seed", 0) or 0)),
            max_depth=int(hole_defaults.get("max_depth", 4) or 4),
            nvars=int(case_state["nvars"]),
            poly_degree=int(hole_defaults.get("poly_degree", 2) or 2),
            preview_topk=int(hole_defaults.get("preview_topk", 4) or 4),
            exact_budget=int(hole_defaults.get("exact_budget", 2) or 2),
            solver_market_enable=True,
            solver_market_preview_topk=int(hole_defaults.get("solver_market_preview_topk", 4) or 4),
            solver_market_exact_topk=int(hole_defaults.get("solver_market_exact_topk", 2) or 2),
            inverse_spec_constant_lift_route_enable=True,
            inverse_spec_constant_lift_route_topk=int(
                hole_defaults.get("constant_lift_route_topk", 2) or 2
            ),
            inverse_spec_coordinate_lift_enable=True,
            inverse_spec_coordinate_lift_topk=int(hole_defaults.get("coordinate_lift_topk", 4) or 4),
            inverse_spec_coordinate_lift_mode=str(hole_defaults.get("coordinate_lift_mode", "both") or "both"),
            inverse_spec_tangent_edit_enable=True,
            inverse_spec_tangent_edit_topk=int(hole_defaults.get("tangent_edit_topk", 8) or 8),
            inverse_spec_soft_edit_enable=True,
            inverse_spec_soft_edit_steps=int(hole_defaults.get("soft_edit_steps", 24) or 24),
            inverse_spec_soft_edit_l1=float(hole_defaults.get("soft_edit_l1", 1.0e-3) or 1.0e-3),
            inverse_spec_witness_loss_enable=bool(hole_defaults.get("witness_loss_enable", True)),
            inverse_spec_witness_grad_weight=float(hole_defaults.get("witness_grad_weight", 0.5) or 0.5),
            inverse_spec_witness_diag_weight=float(hole_defaults.get("witness_diag_weight", 0.25) or 0.25),
            inverse_spec_directional_market_enable=bool(hole_defaults.get("directional_market_enable", True)),
            return_meta=True,
        )

    selected_route = str(meta.get("hole_search_solver_market_selected_route", "") or "")
    selected_subroute = str(meta.get("hole_search_solver_market_selected_subroute", "") or "")
    selected_preview_probe_mse = _safe_float(meta.get("hole_search_selected_preview_probe_mse", None))
    route_match = bool(selected_route in tuple(case_state.get("expected_route_names", ())))
    preview_match = True
    if selected_preview_probe_mse_max is not None:
        preview_match = bool(
            selected_preview_probe_mse is not None
            and selected_preview_probe_mse <= float(selected_preview_probe_mse_max)
        )
    row = {
        "case_id": str(case_state["case_id"]),
        "archetype": str(case_state["archetype"]),
        "expected_route_names": list(case_state.get("expected_route_names", ())),
        "selected_route": selected_route,
        "selected_subroute": selected_subroute,
        "selected_expr": _jsonable(expr),
        "selected_expr_str": "" if expr is None else str(node_str(expr)),
        "selected_preview_probe_mse": selected_preview_probe_mse,
        "selected_preview_probe_mse_max": selected_preview_probe_mse_max,
        "selected_best_eff_mse": _safe_float(meta.get("hole_search_best_eff_mse", None)),
        "route_match": bool(route_match),
        "preview_match": bool(preview_match),
        "success": bool(str(meta.get("status", "")) == "ok" and route_match and preview_match),
        "status": str(meta.get("status", "") or ""),
        "route_error_names": [
            str(route.get("route_name", "") or "")
            for route in list(meta.get("hole_search_solver_market_routes", []) or [])
            if str(route.get("error", "") or "")
        ],
        "route_status_by_name": {
            str(route.get("route_name", "") or ""): str(route.get("status", "") or "")
            for route in list(meta.get("hole_search_solver_market_routes", []) or [])
            if isinstance(route, Mapping)
        },
        "wall_seconds": _safe_float(meta.get("hole_search_wall_seconds", None)),
    }
    if save_dir is not None:
        _write_json(
            {
                "case": {key: value for key, value in case_state.items() if key not in {"x_fit", "t_fit", "x_probe", "t_probe", "opportunity"}},
                "row": row,
                "meta": meta,
            },
            save_dir / f"{str(case_state['case_id'])}.json",
        )
    return row


def _summarize_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    payload_rows = [dict(row) for row in list(rows or []) if isinstance(row, Mapping)]
    if not payload_rows:
        return {
            "n_rows": 0,
            "success_rate": 0.0,
            "route_match_rate": 0.0,
            "preview_match_rate": 0.0,
            "by_archetype": [],
        }
    success_rate = sum(1.0 for row in payload_rows if bool(row.get("success", False))) / len(payload_rows)
    route_match_rate = sum(1.0 for row in payload_rows if bool(row.get("route_match", False))) / len(payload_rows)
    preview_match_rate = sum(1.0 for row in payload_rows if bool(row.get("preview_match", False))) / len(payload_rows)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in payload_rows:
        groups[str(row.get("archetype", "") or "")].append(row)
    by_archetype: list[dict[str, Any]] = []
    for archetype, group_rows in sorted(groups.items()):
        probe_values = [
            float(row["selected_preview_probe_mse"])
            for row in group_rows
            if _safe_float(row.get("selected_preview_probe_mse", None)) is not None
        ]
        by_archetype.append(
            {
                "archetype": archetype,
                "n_rows": int(len(group_rows)),
                "success_rate": float(
                    sum(1.0 for row in group_rows if bool(row.get("success", False))) / len(group_rows)
                ),
                "route_match_rate": float(
                    sum(1.0 for row in group_rows if bool(row.get("route_match", False))) / len(group_rows)
                ),
                "preview_match_rate": float(
                    sum(1.0 for row in group_rows if bool(row.get("preview_match", False))) / len(group_rows)
                ),
                "mean_selected_preview_probe_mse": (
                    float(sum(probe_values) / len(probe_values)) if probe_values else float("nan")
                ),
            }
        )
    return {
        "n_rows": int(len(payload_rows)),
        "success_rate": float(success_rate),
        "route_match_rate": float(route_match_rate),
        "preview_match_rate": float(preview_match_rate),
        "by_archetype": by_archetype,
    }


def run_hole_search_archetype_suite(
    manifest: Mapping[str, Any],
    *,
    manifest_path: pathlib.Path,
    output_dir: pathlib.Path,
    save_individual_reports: bool = False,
) -> dict[str, Any]:
    defaults = dict(manifest.get("defaults", {}) or {})
    suite_id = str(manifest.get("suite_id", "hole_search_archetypes") or "hole_search_archetypes")
    output_dir.mkdir(parents=True, exist_ok=True)
    save_dir = output_dir / "cases" if save_individual_reports else None
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for raw_case in list(manifest.get("cases") or []):
        case = dict(raw_case or {})
        archetype = str(case.get("archetype", "") or "").strip().lower()
        builder = _CASE_BUILDERS.get(archetype, None)
        if builder is None:
            raise ValueError(f"Unknown hole-search archetype {archetype!r} in {case.get('case_id', '')}")
        case_state = builder(case, defaults)
        rows.append(_run_case(case_state, defaults=defaults, save_dir=save_dir))

    summary = _summarize_rows(rows)
    payload = {
        "mode": "hole_search_archetype_benchmark",
        "suite_id": suite_id,
        "suite_manifest": str(manifest_path),
        "n_cases": int(len(list(manifest.get("cases") or []))),
        "rows": rows,
        "summary": summary,
    }
    _write_json(payload, output_dir / "hole_search_archetype_benchmark_results.json")
    _write_json({"suite_id": suite_id, "summary": summary}, output_dir / "hole_search_archetype_benchmark_summary.json")
    return payload


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tiny actual-hole_search archetype regression benchmark")
    parser.add_argument("--suite_manifest", type=str, default=str(DEFAULT_SUITE_MANIFEST))
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--save_individual_reports", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    manifest_path, manifest = load_hole_search_archetype_suite(args.suite_manifest)
    suite_id = str(manifest.get("suite_id", "hole_search_archetypes") or "hole_search_archetypes")
    output_dir = pathlib.Path(
        args.output_dir or (REPO_ROOT / "results" / f"hole_search_archetype_benchmark_{suite_id}")
    )
    payload = run_hole_search_archetype_suite(
        manifest,
        manifest_path=manifest_path,
        output_dir=output_dir,
        save_individual_reports=bool(args.save_individual_reports),
    )
    print(
        f"[hole_search_archetype_benchmark] suite={payload['suite_id']} "
        f"cases={payload['n_cases']} rows={len(payload['rows'])} output_dir={output_dir}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
