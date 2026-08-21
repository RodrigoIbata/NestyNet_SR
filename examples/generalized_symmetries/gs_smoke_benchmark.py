# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch

from nestynet_sr.sr_core import ast_to_human_readable
from nestynet_sr.sr_core.bridges import AtomNode, Mul, Pow, U, Var
from nestynet_sr.sr_gs import GeneralizedSymmetryConfig, discover_generator_specs
from nestynet_sr.sr_gs.prolongation import AffinePointGenerator, score_affine_point_generators_from_jets
from nestynet_sr.sr_search.search import _detect_compound_variable_for_atom


def _noise(a: np.ndarray, rel: float, rng: np.random.Generator) -> np.ndarray:
    scale = float(np.sqrt(np.mean(np.asarray(a, dtype=float) ** 2)))
    return np.asarray(a, dtype=float) + float(rel) * max(scale, 1.0e-12) * rng.normal(size=np.asarray(a).shape)


def _case_bank(n: int, seed: int):
    rng = np.random.default_rng(seed)
    Xn = rng.normal(size=(n, 2))
    Xp = rng.uniform(0.5, 2.0, size=(n, 2))
    r2 = np.sum(Xn * Xn, axis=1)
    zratio = Xp[:, 0] / Xp[:, 1]
    zdiff = Xn[:, 0] - Xn[:, 1]
    zlor = Xn[:, 0] ** 2 - Xn[:, 1] ** 2
    zob = math.sqrt(2.0) * Xn[:, 0] - Xn[:, 1]
    e0, e1 = np.exp(Xn[:, 0]), np.exp(2.0 * Xn[:, 1])
    generic_y = Xn[:, 0] ** 2 + Xn[:, 0] * Xn[:, 1] + 0.7 * Xn[:, 1] ** 3
    generic_g = np.stack([2.0 * Xn[:, 0] + Xn[:, 1], Xn[:, 0] + 2.1 * Xn[:, 1] ** 2], axis=1)
    common = dict(enabled=True, residual_tol=0.03, audit_residual_tol=0.10, min_confidence=0.65)
    return [
        dict(name="radial_SO2", X=Xn, y=np.sin(r2), G=np.stack([2 * Xn[:, 0] * np.cos(r2), 2 * Xn[:, 1] * np.cos(r2)], 1), cfg=GeneralizedSymmetryConfig(**common, translations=False, diagonal_translations=False, scalings=False, rotations=True), expected=lambda s: s.family == "rotation" and s.kind == "so2_pair"),
        dict(name="ratio_scaling", X=Xp, y=np.sin(zratio), G=np.stack([np.cos(zratio) / Xp[:, 1], -Xp[:, 0] * np.cos(zratio) / Xp[:, 1] ** 2], 1), cfg=GeneralizedSymmetryConfig(**common, translations=False, diagonal_translations=False, scalings=True, rotations=False), expected=lambda s: s.family == "scaling" and s.kind == "common_pair"),
        dict(name="difference_translation", X=Xn, y=zdiff ** 3, G=np.stack([3 * zdiff ** 2, -3 * zdiff ** 2], 1), cfg=GeneralizedSymmetryConfig(**common, translations=False, diagonal_translations=True, scalings=False, rotations=False), expected=lambda s: s.family == "translation" and s.kind == "diagonal_plus"),
        dict(name="lorentz_interval", X=Xn, y=np.sin(zlor), G=np.stack([2 * Xn[:, 0] * np.cos(zlor), -2 * Xn[:, 1] * np.cos(zlor)], 1), cfg=GeneralizedSymmetryConfig(**common, translations=False, diagonal_translations=False, scalings=False, rotations=False, lorentz_boosts=True), expected=lambda s: s.family == "lorentz" and s.kind == "boost_pair"),
        dict(name="learned_oblique_translation", X=Xn, y=np.sin(zob), G=np.stack([math.sqrt(2.0) * np.cos(zob), -np.cos(zob)], 1), cfg=GeneralizedSymmetryConfig(**common, known_generators=False, known_lie=False, general_affine=True, translations=False, diagonal_translations=False, scalings=False, rotations=False, output_equivariance=False), expected=lambda s: s.family == "general_affine" and s.kind == "affine_translation_pair"),
        dict(name="affine_output_equivariance", X=Xn, y=e0 + e1, G=np.stack([e0, 2.0 * e1], 1), cfg=GeneralizedSymmetryConfig(**common, known_generators=False, known_lie=False, general_affine=True, translations=False, diagonal_translations=False, scalings=False, rotations=False, output_equivariance=True), expected=lambda s: s.family == "general_affine" and s.kind == "affine_translation_pair" and abs(s.output_beta) > 0.5),
        dict(name="generic_negative_control", X=Xn, y=generic_y, G=generic_g, cfg=GeneralizedSymmetryConfig(**common, known_generators=True, known_lie=True, general_affine=True, lorentz_boosts=True, output_equivariance=False), expected=lambda s: False),
    ]


def run_sr_noise_benchmark(n: int, repeats: int, seed: int):
    levels = (0.0, 1.0e-4, 1.0e-3, 1.0e-2, 3.0e-2, 1.0e-1)
    rows = []
    for case_index, case in enumerate(_case_bank(n, seed)):
        for rel in levels:
            hits, residuals, accepted_counts = 0, [], []
            for rep in range(repeats):
                rng = np.random.default_rng(seed + 10000 * case_index + 101 * rep + int(rel * 1.0e7))
                y = _noise(case["y"], rel, rng)
                G = _noise(case["G"], rel, rng)
                specs = discover_generator_specs(case["X"], y, G, cfg=case["cfg"])
                accepted_counts.append(len(specs))
                matches = [s for s in specs if case["expected"](s)]
                if matches:
                    hits += 1
                    residuals.append(min(float(s.residual_rel) for s in matches))
            is_negative = case["name"] == "generic_negative_control"
            rows.append({
                "case": case["name"],
                "relative_noise": rel,
                "detection_rate": hits / repeats,
                "false_positive_rate": float(np.mean(np.asarray(accepted_counts) > 0)) if is_negative else None,
                "median_expected_residual": float(np.median(residuals)) if residuals else None,
                "median_accepted_count": float(np.median(accepted_counts)),
            })
    return rows


def _angle_residual(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(math.sqrt(max(0.0, 1.0 - float(np.dot(a, b) ** 2 / (np.dot(a, a) * np.dot(b, b))))))


def _linear_coeffs_from_text(text: Any, nvars: int = 2):
    raw = str(text or "")
    if not raw:
        return None
    coeffs = np.zeros(int(nvars), dtype=float)
    hit = False
    for match in re.finditer(r"([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*\*\s*x(\d+)", raw):
        axis = int(match.group(2))
        if axis < int(nvars):
            coeffs[axis] += float(match.group(1))
            hit = True
    stripped = re.sub(r"([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*\*\s*x\d+", " ", raw)
    for match in re.finditer(r"(?<![A-Za-z0-9_])([+-]?)\s*x(\d+)", stripped):
        axis = int(match.group(2))
        if axis < int(nvars):
            sign = -1.0 if match.group(1) == "-" else 1.0
            coeffs[axis] += sign
            hit = True
    return coeffs if hit else None


def run_stagea_oblique(seed: int):
    class ObliqueLeaf(torch.nn.Module):
        def forward(self, x):
            return torch.sin(math.sqrt(2.0) * x[:, 0:1] - x[:, 1:2])
        def grad(self, cache):
            x = cache["x"]
            c = torch.cos(math.sqrt(2.0) * x[:, 0] - x[:, 1])
            return torch.stack([math.sqrt(2.0) * c, -c], dim=1).unsqueeze(1)

    rng = np.random.default_rng(seed)
    x = torch.tensor(rng.normal(size=(512, 2)), dtype=torch.float64)
    y = torch.zeros((512, 1), dtype=torch.float64)
    atom = AtomNode(kind="nn", var_idxs=(0, 1), kwargs={"num_segments": 8, "dual_layer": False}, tag="nn_oblique")
    common = dict(model=object(), atom=atom, leaf=ObliqueLeaf(), datagen_train=[(x, y)], device=torch.device("cpu"), max_batches=1, enable_shift=False, enable_mixed_compound=False, trig_axis_specs=None, scaling_features=None, invariance_features=None)
    cfg = GeneralizedSymmetryConfig(enabled=True, known_generators=False, known_lie=False, general_affine=True, translations=False, diagonal_translations=False, scalings=False, rotations=False, output_equivariance=False, residual_tol=1.0e-8, audit_residual_tol=1.0e-6, min_confidence=0.5)
    out = {}
    for label, gs_cfg in (("baseline", None), ("gs", cfg)):
        with contextlib.redirect_stdout(io.StringIO()):
            proposals, _ = _detect_compound_variable_for_atom(**common, gs_cfg=gs_cfg)
        rows = []
        for p in proposals:
            meta = p[4] or {}
            expr = ast_to_human_readable(p[1])
            z_human = meta.get("z_human") or expr
            inv_coeffs = _linear_coeffs_from_text(z_human)
            rows.append(
                {
                    "pattern": list(p[0]),
                    "expression": expr,
                    "confidence": float(p[2]),
                    "source": meta.get("source"),
                    "kind": meta.get("kind"),
                    "gs_kind": meta.get("gs_kind"),
                    "generator_coeffs": list(meta.get("gs_generator_coeffs", ())),
                    "invariant_coeffs": inv_coeffs.tolist() if inv_coeffs is not None else [],
                }
            )
        out[label] = rows
    true_coordinate = np.asarray([math.sqrt(2.0), -1.0])
    baseline = next((p for p in out["baseline"] if p["kind"] == "linear"), None)
    gs = next((p for p in out["gs"] if p["source"] == "generalized_symmetry"), None)
    baseline_vec = np.asarray(baseline["pattern"], float) if baseline else None
    b = np.asarray(gs["generator_coeffs"], float) if gs else None
    gs_vec = np.asarray([b[1], -b[0]]) if b is not None and b.size == 2 else None
    if gs_vec is None and gs:
        coeffs = np.asarray(gs.get("invariant_coeffs", []), float)
        if coeffs.size == 2:
            gs_vec = coeffs
    return {
        "target": "sin(sqrt(2)*x0-x1)",
        "baseline_proposals": out["baseline"],
        "gs_proposals": out["gs"],
        "baseline_coordinate_angle_residual": _angle_residual(baseline_vec, true_coordinate) if baseline_vec is not None else None,
        "gs_coordinate_angle_residual": _angle_residual(gs_vec, true_coordinate) if gs_vec is not None else None,
    }


def _generator_row(meta: dict[str, Any], name: str):
    return next(row for row in meta["generators"] if row.get("name") == name)


def run_de_benchmark():
    x = torch.linspace(0.2, 3.0, 192, dtype=torch.float64).unsqueeze(1)
    u, u1, u2 = torch.sin(x), torch.cos(x), -torch.sin(x)
    ho = score_affine_point_generators_from_jets(order=2, x=x, u=u, u1=u1, u2=u2, term_asts=[U()], coeffs=torch.tensor([1.0], dtype=torch.float64), include_known=True, include_general_affine=False, tol=1.0e-8)
    nonhom = score_affine_point_generators_from_jets(order=2, x=x, u=u, u1=u1, u2=u2, term_asts=[U(), None], coeffs=torch.tensor([1.0, 1.0], dtype=torch.float64), include_known=True, include_general_affine=False, tol=0.05)
    nonsym_term = Mul(Var(0), U())
    scale_rows = []
    for scale in (1.0, 0.1, 1.0e-6):
        meta = score_affine_point_generators_from_jets(order=2, x=x, u=u, u1=u1, u2=u2, term_asts=[nonsym_term], coeffs=torch.tensor([1.0], dtype=torch.float64), generators=[AffinePointGenerator("x_translation", "translation", a0=scale)], tol=0.05)
        row = meta["generators"][0]
        scale_rows.append({"generator_scale": scale, "metric": row["on_shell_metric"], "accepted": row["accepted"]})
    xr = torch.linspace(0.2, 4.0, 192, dtype=torch.float64).unsqueeze(1)
    ur, ur1, ur2 = 1.0 / xr, -1.0 / xr.pow(2), 2.0 / xr.pow(3)
    radial = score_affine_point_generators_from_jets(order=1, x=xr, u=ur, u1=ur1, u2=ur2, term_asts=[Mul(Pow(Var(0), -1.0), U())], coeffs=torch.tensor([1.0], dtype=torch.float64), include_known=True, include_general_affine=False, tol=1.0e-8)
    return {
        "harmonic_oscillator": {
            "tested_generators": ho["tested_generators"],
            "accepted_count": len(ho["accepted_generator_names"]),
            **{name: {"metric": _generator_row(ho, name)["on_shell_metric"], "accepted": _generator_row(ho, name)["accepted"]} for name in ("x_translation", "u_scaling")},
        },
        "nonhomogeneous_oscillator": {
            "tested_generators": nonhom["tested_generators"],
            "accepted_count": len(nonhom["accepted_generator_names"]),
            **{name: {"metric": _generator_row(nonhom, name)["on_shell_metric"], "accepted": _generator_row(nonhom, name)["accepted"]} for name in ("x_translation", "u_scaling")},
        },
        "radial_first_order": {
            "tested_generators": radial["tested_generators"],
            "accepted_count": len(radial["accepted_generator_names"]),
            "x_scaling": {"metric": _generator_row(radial, "x_scaling")["on_shell_metric"], "accepted": _generator_row(radial, "x_scaling")["accepted"]},
        },
        "projective_scale_control": scale_rows,
    }


def _finite_or_none(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def _zero_noise_rows(noise_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in noise_rows:
        if _finite_or_none(row.get("relative_noise")) == 0.0:
            out[str(row.get("case"))] = row
    return out


def _row(
    *,
    family: str,
    case: str,
    arm: str,
    metric: str,
    value: Any,
    complexity_metric: str,
    complexity_value: Any,
    efficiency_metric: str,
    efficiency_value: Any,
    claim_tier: str,
    interpretation: str,
) -> dict[str, Any]:
    return {
        "benchmark_family": family,
        "case": case,
        "arm": arm,
        "final_accuracy_metric": metric,
        "final_accuracy_value": _finite_or_none(value),
        "complexity_metric": complexity_metric,
        "complexity_value": _finite_or_none(complexity_value),
        "search_efficiency_metric": efficiency_metric,
        "search_efficiency_value": _finite_or_none(efficiency_value),
        "claim_tier": claim_tier,
        "interpretation": interpretation,
    }


def build_paper_summary(payload: dict[str, Any]) -> dict[str, Any]:
    """Build a conservative paper-facing summary from raw smoke results."""

    rows: list[dict[str, Any]] = []
    zero = _zero_noise_rows(list(payload.get("sr_noise", []) or []))
    detector_cases = {
        "radial_SO2": ("current detector-style GS", "named rotation detects an invariant radial coordinate; compare against any baseline that already includes r^2"),
        "ratio_scaling": ("current detector-style GS", "named scaling detects a ratio invariant; matched libraries with x0/x1 must be reported separately"),
        "difference_translation": ("current detector-style GS", "named diagonal translation detects a difference invariant"),
        "lorentz_interval": ("current detector-style GS", "named graph Lorentz detector finds an interval-style invariant; this is not PDE/vector Lorentz support"),
    }
    for case, (arm, note) in detector_cases.items():
        raw = zero.get(case, {})
        rows.append(
            _row(
                family="analytic_sr_generator_detection",
                case=case,
                arm=arm,
                metric="zero_noise_detection_rate",
                value=raw.get("detection_rate"),
                complexity_metric="median_accepted_generators",
                complexity_value=raw.get("median_accepted_count"),
                efficiency_metric="relative_noise_levels",
                efficiency_value=len({r.get("relative_noise") for r in payload.get("sr_noise", []) if r.get("case") == case}),
                claim_tier="detector_recovery",
                interpretation=note,
            )
        )

    oblique = zero.get("learned_oblique_translation", {})
    rows.append(
        _row(
            family="analytic_sr_generator_detection",
            case="learned_oblique_translation",
            arm="global affine algebra",
            metric="zero_noise_detection_rate",
            value=oblique.get("detection_rate"),
            complexity_metric="median_accepted_generators",
            complexity_value=oblique.get("median_accepted_count"),
            efficiency_metric="relative_noise_levels",
            efficiency_value=len({r.get("relative_noise") for r in payload.get("sr_noise", []) if r.get("case") == "learned_oblique_translation"}),
            claim_tier="coordinate_discovery",
            interpretation="noninteger oblique quotient direction absent from the integer-affine baseline vocabulary",
        )
    )

    out_link = zero.get("affine_output_equivariance", {})
    rows.append(
        _row(
            family="analytic_sr_generator_detection",
            case="affine_output_equivariance",
            arm="output-link GS",
            metric="zero_noise_detection_rate",
            value=out_link.get("detection_rate"),
            complexity_metric="median_accepted_generators",
            complexity_value=out_link.get("median_accepted_count"),
            efficiency_metric="relative_noise_levels",
            efficiency_value=len({r.get("relative_noise") for r in payload.get("sr_noise", []) if r.get("case") == "affine_output_equivariance"}),
            claim_tier="normal_form_discovery",
            interpretation="detects affine-output equivariance; downstream expression claims require the normal-form path to use the witness",
        )
    )

    negative = zero.get("generic_negative_control", {})
    rows.append(
        _row(
            family="negative_control",
            case="generic_negative_control",
            arm="all GS detectors",
            metric="zero_noise_false_positive_rate",
            value=negative.get("false_positive_rate"),
            complexity_metric="median_accepted_generators",
            complexity_value=negative.get("median_accepted_count"),
            efficiency_metric="relative_noise_levels",
            efficiency_value=len({r.get("relative_noise") for r in payload.get("sr_noise", []) if r.get("case") == "generic_negative_control"}),
            claim_tier="negative_control",
            interpretation="guards against treating every low-complexity coordinate as a symmetry",
        )
    )

    stagea = payload.get("stagea_oblique", {}) or {}
    baseline_props = len(stagea.get("baseline_proposals", []) or [])
    gs_props = len(stagea.get("gs_proposals", []) or [])
    rows.extend(
        [
            _row(
                family="stagea_coordinate_rewrite",
                case="sin(sqrt(2)*x0-x1)",
                arm="baseline integer compound detector",
                metric="coordinate_angle_residual",
                value=stagea.get("baseline_coordinate_angle_residual"),
                complexity_metric="proposal_count",
                complexity_value=baseline_props,
                efficiency_metric="proposal_count",
                efficiency_value=baseline_props,
                claim_tier="baseline_control",
                interpretation="baseline detector is useful but cannot exactly express the irrational oblique coordinate",
            ),
            _row(
                family="stagea_coordinate_rewrite",
                case="sin(sqrt(2)*x0-x1)",
                arm="global affine quotient GS",
                metric="coordinate_angle_residual",
                value=stagea.get("gs_coordinate_angle_residual"),
                complexity_metric="proposal_count",
                complexity_value=gs_props,
                efficiency_metric="proposal_count",
                efficiency_value=gs_props,
                claim_tier="coordinate_discovery",
                interpretation="direct evidence for better coordinates rather than a larger hard-coded library",
            ),
        ]
    )

    de = payload.get("de_prolongation", {}) or {}
    ho = de.get("harmonic_oscillator", {}) or {}
    nonhom = de.get("nonhomogeneous_oscillator", {}) or {}
    radial = de.get("radial_first_order", {}) or {}
    rows.extend(
        [
            _row(
                family="de_relative_invariance",
                case="harmonic_oscillator",
                arm="DE point-prolongation certificate",
                metric="u_scaling_on_shell_metric",
                value=(ho.get("u_scaling") or {}).get("metric"),
                complexity_metric="accepted_generators",
                complexity_value=ho.get("accepted_count"),
                efficiency_metric="tested_generators",
                efficiency_value=ho.get("tested_generators"),
                claim_tier="equation_level_certificate",
                interpretation="proves a residual-level generator on the scalar ODE candidate; audit by default",
            ),
            _row(
                family="de_relative_invariance",
                case="nonhomogeneous_oscillator",
                arm="DE point-prolongation rejection",
                metric="u_scaling_on_shell_metric",
                value=(nonhom.get("u_scaling") or {}).get("metric"),
                complexity_metric="accepted_generators",
                complexity_value=nonhom.get("accepted_count"),
                efficiency_metric="tested_generators",
                efficiency_value=nonhom.get("tested_generators"),
                claim_tier="negative_control",
                interpretation="same solution jets with a nonhomogeneous residual reject output scaling",
            ),
            _row(
                family="de_invariant_library_control",
                case="radial_first_order",
                arm="scalar DE invariant/prior row",
                metric="x_scaling_on_shell_metric",
                value=(radial.get("x_scaling") or {}).get("metric"),
                complexity_metric="accepted_generators",
                complexity_value=radial.get("accepted_count"),
                efficiency_metric="tested_generators",
                efficiency_value=radial.get("tested_generators"),
                claim_tier="library_prior_requires_matched_control",
                interpretation="radial rows are useful but must be compared with neutral vocabulary arms before claiming GS advantage",
            ),
        ]
    )

    return {
        "schema": "nestynet_sr_gs_geometry_paper_summary_v1",
        "headline_message": "Generalized symmetries are a coordinate and normal-form discovery layer; detector menus are implementation details.",
        "benchmark_rows": rows,
        "matched_library_controls": [
            "baseline versus neutral hard-tail vocabulary isolates manual vocabulary expansion",
            "GS audit-only should preserve baseline search behavior",
            "GS proposal rows should be reported separately from explicit selection-time scoring",
            "oracle or target-specific templates are debugging upper bounds, not GS headline evidence",
        ],
        "limitations": [
            "scalar ODE prolongation is implemented; vector/PDE jet spaces are represented but unsupported for scoring",
            "DE hard-tail and invariant-library rows are source-aware structural priors unless certified by equation-level residual tests",
            "unit-torus results require scientifically justified units and matched no-units controls",
            "smoke benchmarks are deterministic integration evidence, not final paper-scale accuracy tables",
        ],
    }


def _fmt_cell(value: Any) -> str:
    value_f = _finite_or_none(value)
    if value_f is None:
        return ""
    if abs(value_f) >= 1.0e4 or (abs(value_f) < 1.0e-3 and value_f != 0.0):
        return f"{value_f:.3e}"
    return f"{value_f:.6g}"


def write_paper_markdown(summary: dict[str, Any], path: Path) -> None:
    rows = list(summary.get("benchmark_rows", []) or [])
    lines = [
        "# GS Geometry Benchmark Framing",
        "",
        summary.get("headline_message", ""),
        "",
        "| family | case | arm | final metric | value | complexity | search | claim tier |",
        "|---|---|---|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {family} | `{case}` | {arm} | {metric} | {value} | {complexity} | {search} | `{tier}` |".format(
                family=row["benchmark_family"],
                case=row["case"],
                arm=row["arm"],
                metric=row["final_accuracy_metric"],
                value=_fmt_cell(row["final_accuracy_value"]),
                complexity=f"{row['complexity_metric']}={_fmt_cell(row['complexity_value'])}",
                search=f"{row['search_efficiency_metric']}={_fmt_cell(row['search_efficiency_value'])}",
                tier=row["claim_tier"],
            )
        )
    lines.extend(["", "## Matched Controls", ""])
    lines.extend(f"- {item}" for item in summary.get("matched_library_controls", []) or [])
    lines.extend(["", "## Limitations", ""])
    lines.extend(f"- {item}" for item in summary.get("limitations", []) or [])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, default=Path("gs_smoke_results.json"))
    ap.add_argument("--csv", type=Path, default=Path("gs_noise_results.csv"))
    ap.add_argument("--markdown", type=Path, default=None, help="Optional paper-facing Markdown summary path")
    ap.add_argument("--samples", type=int, default=1024)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--seed", type=int, default=20260621)
    args = ap.parse_args()
    noise_rows = run_sr_noise_benchmark(args.samples, args.repeats, args.seed)
    payload = {"samples": args.samples, "repeats": args.repeats, "seed": args.seed, "sr_noise": noise_rows, "stagea_oblique": run_stagea_oblique(args.seed + 1), "de_prolongation": run_de_benchmark()}
    payload["paper_summary"] = build_paper_summary(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    with args.csv.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(noise_rows[0]))
        writer.writeheader(); writer.writerows(noise_rows)
    if args.markdown is not None:
        write_paper_markdown(payload["paper_summary"], args.markdown)
    print(json.dumps({"json": str(args.output), "csv": str(args.csv), "markdown": str(args.markdown) if args.markdown else None, "sr_rows": len(noise_rows), "paper_rows": len(payload["paper_summary"]["benchmark_rows"])}, indent=2))


if __name__ == "__main__":
    main()
