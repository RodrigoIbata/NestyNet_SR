#!/usr/bin/env python3
# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
import ssl
from typing import Any
import urllib.parse
import urllib.request

import pandas as pd

JPL_SBDB_QUERY_API_URL = "https://ssd-api.jpl.nasa.gov/sbdb_query.api"
JPL_SBDB_OBJECT_API_URL = "https://ssd-api.jpl.nasa.gov/sbdb.api"
SSODNET_MAIN_BELT_CLASSES = ("MB>Inner", "MB>Middle", "MB>Outer")

# The live upstream catalogs currently need two explicit exceptions:
# - (1) Ceres is omitted by the JPL bulk query despite the direct SBDB object API
#   classifying it as an MBA with belt-like geometry.
# - (15) Eunomia has a malformed SsODNet parquet mass entry (`4.4341`) that is
#   inconsistent with independent literature estimates of order 3e19 kg.
#   We therefore override its mass with a documented manual value anchored to
#   independent sources (SiMDA and Gaia astrometric mass work).
MANUAL_OBJECT_OVERRIDES: dict[int, dict[str, Any]] = {
    1: {
        "selection_note": (
            "Manual add-back: missing from the JPL bulk candidate query even though the "
            "direct JPL SBDB object API classifies it as a main-belt asteroid with "
            "belt-like orbital geometry."
        ),
    },
    15: {
        "mass_override_kg": 3.025e19,
        "selection_note": (
            "Manual mass override: SsODNet parquet mass.value is malformed (`4.4341`) "
            "for Eunomia. Replaced with 3.025e19 kg, consistent with independent "
            "literature estimates (SiMDA mean and Gaia astrometric mass results)."
        ),
    },
}


def _fetch_json(url: str, *, verify_ssl: bool) -> dict[str, Any]:
    context = None if bool(verify_ssl) else ssl._create_unverified_context()
    with urllib.request.urlopen(url, timeout=120, context=context) as response:
        return json.loads(response.read().decode("utf-8"))


def _slugify_name(text: str) -> str:
    out = re.sub(r"[^A-Za-z0-9]+", "_", str(text).strip().lower())
    out = out.strip("_")
    return out or "unnamed"


def _orbit_id(number: int, name: str) -> str:
    return f"mp_{int(number)}_{_slugify_name(name)}"


def _parse_manual_numbers(raw: str) -> list[int]:
    parts = [part.strip() for part in str(raw).split(",")]
    return [int(part) for part in parts if part]


def _query_jpl_candidates(
    *,
    a_min: float,
    a_max: float,
    q_min: float,
    data_arc_min: int,
    n_obs_min: int,
    limit: int,
    verify_ssl: bool,
) -> tuple[list[dict[str, Any]], int]:
    constraints = {
        "AND": [
            f"a|GE|{float(a_min):g}",
            f"a|LE|{float(a_max):g}",
            f"q|GT|{float(q_min):g}",
            f"data_arc|GE|{int(data_arc_min)}",
            f"n_obs_used|GE|{int(n_obs_min)}",
        ]
    }
    params = {
        "fields": "pdes,full_name,a,q,ad,data_arc,n_obs_used",
        "sb-ns": "n",
        "limit": str(int(limit)),
        "sb-cdata": json.dumps(constraints, separators=(",", ":")),
    }
    url = JPL_SBDB_QUERY_API_URL + "?" + urllib.parse.urlencode(params)
    payload = _fetch_json(url, verify_ssl=verify_ssl)
    rows = list(payload.get("data", []) or [])
    total_count = int(payload.get("count", len(rows)))
    if len(rows) != total_count:
        raise ValueError(
            f"JPL query returned {len(rows)} rows but reported count={total_count}; "
            "increase --jpl_limit"
        )
    fields = list(payload.get("fields", []) or [])
    field_idx = {str(name): idx for idx, name in enumerate(fields)}
    out = []
    for row in rows:
        pdes = str(row[field_idx["pdes"]]).strip()
        if not pdes.isdigit():
            continue
        out.append(
            {
                "sso_number": int(pdes),
                "jpl_full_name": str(row[field_idx["full_name"]]).strip(),
                "a": float(row[field_idx["a"]]),
                "q": float(row[field_idx["q"]]),
                "ad": float(row[field_idx["ad"]]),
                "data_arc": int(row[field_idx["data_arc"]]),
                "n_obs_used": int(row[field_idx["n_obs_used"]]),
                "selection_source": "jpl_bulk_query",
            }
        )
    return out, total_count


def _query_jpl_object(number: int, *, verify_ssl: bool) -> dict[str, Any]:
    params = {"sstr": str(int(number)), "phys-par": "1"}
    url = JPL_SBDB_OBJECT_API_URL + "?" + urllib.parse.urlencode(params)
    payload = _fetch_json(url, verify_ssl=verify_ssl)
    obj = dict(payload.get("object", {}) or {})
    orbit = dict(payload.get("orbit", {}) or {})
    elements = list(orbit.get("elements", []) or [])
    element_map = {
        str(item["name"]): item["value"]
        for item in elements
        if isinstance(item, dict) and "name" in item and "value" in item
    }
    orbit_class = dict(obj.get("orbit_class", {}) or {})
    return {
        "sso_number": int(number),
        "jpl_full_name": str(obj.get("fullname", f"{number}")).strip(),
        "kind": str(obj.get("kind", "")),
        "orbit_class_code": str(orbit_class.get("code", "")),
        "orbit_class_name": str(orbit_class.get("name", "")),
        "a": float(element_map["a"]),
        "q": float(element_map["q"]),
        "ad": float(element_map["ad"]),
        "data_arc": int(orbit.get("data_arc", 0)),
        "n_obs_used": int(orbit.get("n_obs_used", 0)),
    }


def _effective_mass_info(number: int, row: dict[str, Any]) -> dict[str, Any] | None:
    override = dict(MANUAL_OBJECT_OVERRIDES.get(int(number), {}) or {})
    if "mass_override_kg" in override:
        return {
            "mass_kg": float(override["mass_override_kg"]),
            "mass_source": "manual_override",
            "raw_ssodnet_mass_kg": None if pd.isna(row.get("mass.value", None)) else float(row["mass.value"]),
            "selection_note": str(override.get("selection_note", "")),
        }

    raw_mass = row.get("mass.value", None)
    if raw_mass is None or pd.isna(raw_mass):
        return None
    raw_mass_f = float(raw_mass)
    if not math.isfinite(raw_mass_f):
        return None
    return {
        "mass_kg": raw_mass_f,
        "mass_source": "ssodnet",
        "raw_ssodnet_mass_kg": raw_mass_f,
        "selection_note": str(override.get("selection_note", "")),
    }


def _load_ssodnet_main_belt(
    *,
    parquet_path: Path,
) -> pd.DataFrame:
    cols = [
        "sso_number",
        "sso_name",
        "sso_class",
        "mass.value",
        "orbital_elements.semi_major_axis.value",
        "orbital_elements.periapsis_distance.value",
        "orbital_elements.apoapsis_distance.value",
        "orbital_elements.orbital_arc",
        "orbital_elements.number_observation",
    ]
    df = pd.read_parquet(parquet_path, columns=cols)
    df = df[df["sso_class"].isin(SSODNET_MAIN_BELT_CLASSES)].copy()
    df = df[df["sso_number"].notna()].copy()
    df["sso_number"] = df["sso_number"].astype(int)
    return df


def _meets_numeric_main_belt_rule(
    *,
    a: float,
    q: float,
    a_min: float,
    a_max: float,
    q_min: float,
    data_arc: int,
    n_obs_used: int,
    data_arc_min: int,
    n_obs_min: int,
) -> bool:
    return (
        _meets_numeric_main_belt_geometry(
            a=float(a),
            q=float(q),
            a_min=float(a_min),
            a_max=float(a_max),
            q_min=float(q_min),
        )
        and int(data_arc) >= int(data_arc_min)
        and int(n_obs_used) >= int(n_obs_min)
    )


def _meets_numeric_main_belt_geometry(
    *,
    a: float,
    q: float,
    a_min: float,
    a_max: float,
    q_min: float,
) -> bool:
    return (
        float(a) >= float(a_min)
        and float(a) <= float(a_max)
        and float(q) > float(q_min)
    )


def _build_candidate_rows(
    *,
    ssodnet_df: pd.DataFrame,
    mass_min: float,
    jpl_rows: list[dict[str, Any]],
    manual_include_numbers: list[int],
    a_min: float,
    a_max: float,
    q_min: float,
    data_arc_min: int,
    n_obs_min: int,
    verify_ssl: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    ssodnet_by_number = {
        int(row["sso_number"]): row
        for row in ssodnet_df.to_dict(orient="records")
    }
    selected: dict[int, dict[str, Any]] = {}
    for row in list(jpl_rows):
        number = int(row["sso_number"])
        ssod = ssodnet_by_number.get(number, None)
        if ssod is None:
            continue
        mass_info = _effective_mass_info(number, ssod)
        if mass_info is None or float(mass_info["mass_kg"]) <= float(mass_min):
            continue
        selected[number] = {
            "orbit_id": _orbit_id(number, str(ssod["sso_name"])),
            "body_name": str(ssod["sso_name"]),
            "split": "candidate",
            "horizons_command": f"{number};",
            "sso_number": number,
            "selection_source": str(row["selection_source"]),
            "jpl_full_name": str(row["jpl_full_name"]),
            "sso_class": str(ssod["sso_class"]),
            "mass_kg": float(mass_info["mass_kg"]),
            "mass_source": str(mass_info["mass_source"]),
            "raw_ssodnet_mass_kg": mass_info["raw_ssodnet_mass_kg"],
            "selection_note": str(mass_info["selection_note"]),
            "semi_major_axis_au": float(ssod["orbital_elements.semi_major_axis.value"]),
            "periapsis_distance_au": float(ssod["orbital_elements.periapsis_distance.value"]),
            "apoapsis_distance_au": float(ssod["orbital_elements.apoapsis_distance.value"]),
            "jpl_data_arc_days": int(row["data_arc"]),
            "jpl_n_obs_used": int(row["n_obs_used"]),
            "ssodnet_orbital_arc_days": int(ssod["orbital_elements.orbital_arc"]),
            "ssodnet_n_obs_used": int(ssod["orbital_elements.number_observation"]),
        }

    manual_added = []
    for number in list(manual_include_numbers):
        if int(number) in selected:
            continue
        ssod = ssodnet_by_number.get(int(number), None)
        if ssod is None:
            raise ValueError(f"manual include number {number} is missing from the SsODNet main-belt subset")
        mass_info = _effective_mass_info(int(number), ssod)
        if mass_info is None or float(mass_info["mass_kg"]) <= float(mass_min):
            raise ValueError(
                f"manual include number {number} does not have a usable effective mass above the threshold"
            )
        jpl = _query_jpl_object(int(number), verify_ssl=verify_ssl)
        if not _meets_numeric_main_belt_geometry(
            a=float(jpl["a"]),
            q=float(jpl["q"]),
            a_min=float(a_min),
            a_max=float(a_max),
            q_min=float(q_min),
        ):
            raise ValueError(
                f"manual include number {number} failed the numeric main-belt geometry rule "
                f"(a={jpl['a']}, q={jpl['q']})"
            )
        direct_quality_pass = (
            int(jpl["data_arc"]) >= int(data_arc_min)
            and int(jpl["n_obs_used"]) >= int(n_obs_min)
        )
        row = {
            "orbit_id": _orbit_id(int(number), str(ssod["sso_name"])),
            "body_name": str(ssod["sso_name"]),
            "split": "candidate",
            "horizons_command": f"{int(number)};",
            "sso_number": int(number),
            "selection_source": "manual_include_after_direct_jpl_check",
            "jpl_full_name": str(jpl["jpl_full_name"]),
            "sso_class": str(ssod["sso_class"]),
            "mass_kg": float(mass_info["mass_kg"]),
            "mass_source": str(mass_info["mass_source"]),
            "raw_ssodnet_mass_kg": mass_info["raw_ssodnet_mass_kg"],
            "selection_note": str(mass_info["selection_note"]),
            "semi_major_axis_au": float(ssod["orbital_elements.semi_major_axis.value"]),
            "periapsis_distance_au": float(ssod["orbital_elements.periapsis_distance.value"]),
            "apoapsis_distance_au": float(ssod["orbital_elements.apoapsis_distance.value"]),
            "jpl_data_arc_days": int(jpl["data_arc"]),
            "jpl_n_obs_used": int(jpl["n_obs_used"]),
            "jpl_orbit_class_code": str(jpl["orbit_class_code"]),
            "jpl_orbit_class_name": str(jpl["orbit_class_name"]),
            "ssodnet_orbital_arc_days": int(ssod["orbital_elements.orbital_arc"]),
            "ssodnet_n_obs_used": int(ssod["orbital_elements.number_observation"]),
            "manual_quality_override": bool(not direct_quality_pass),
            "manual_quality_override_reason": (
                None
                if direct_quality_pass
                else (
                    "direct JPL object query failed the quality gate, "
                    "but the object was manually restored after geometric belt verification"
                )
            ),
        }
        selected[int(number)] = row
        manual_added.append(
            {
                "sso_number": int(number),
                "body_name": str(ssod["sso_name"]),
                "reason": (
                    "missing from JPL bulk candidate query; restored by manual override "
                    "after direct JPL geometric belt verification"
                ),
                "mass_source": str(mass_info["mass_source"]),
                "selection_note": str(mass_info["selection_note"]),
                "direct_jpl_data_arc_days": int(jpl["data_arc"]),
                "direct_jpl_n_obs_used": int(jpl["n_obs_used"]),
                "ssodnet_orbital_arc_days": int(ssod["orbital_elements.orbital_arc"]),
                "ssodnet_n_obs_used": int(ssod["orbital_elements.number_observation"]),
            }
        )

    out_rows = sorted(
        list(selected.values()),
        key=lambda row: (-float(row["mass_kg"]), int(row["sso_number"])),
    )
    return out_rows, manual_added


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build a reproducible candidate manifest for massive main-belt asteroids "
            "by intersecting JPL SBDB numerical orbital cuts with SsODNet mass estimates"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--ssodnet_parquet", type=str, required=True, help="Path to the downloaded SsODNet asteroid ssoBFT parquet")
    parser.add_argument(
        "--output_manifest",
        type=str,
        default=str(Path(__file__).resolve().parent / "data" / "horizons_sources_jpl_ssodnet_mass_gt_1e17_arc15000.json"),
        help="JSON manifest of candidate bodies suitable for fetch_horizons_vectors.py",
    )
    parser.add_argument(
        "--output_summary",
        type=str,
        default=str(Path(__file__).resolve().parent / "data" / "selection_jpl_ssodnet_mass_gt_1e17_arc15000_summary.json"),
        help="JSON summary of the selection rule and candidate rows",
    )
    parser.add_argument("--mass_min", type=float, default=1.0e17, help="Minimum SsODNet mass estimate in kg")
    parser.add_argument("--a_min", type=float, default=2.1, help="Minimum semi-major axis in AU")
    parser.add_argument("--a_max", type=float, default=3.3, help="Maximum semi-major axis in AU")
    parser.add_argument("--q_min", type=float, default=1.6, help="Strict minimum perihelion distance in AU")
    parser.add_argument("--data_arc_min", type=int, default=15000, help="Minimum JPL orbital arc in days")
    parser.add_argument("--n_obs_min", type=int, default=200, help="Minimum JPL observation count")
    parser.add_argument("--jpl_limit", type=int, default=50000, help="Row limit passed to the JPL SBDB query API")
    parser.add_argument(
        "--manual_include_numbers",
        type=str,
        default="1",
        help="Comma-separated object numbers to add back after direct JPL verification (default keeps Ceres)",
    )
    parser.add_argument(
        "--verify_ssl",
        action="store_true",
        help="Verify HTTPS certificates; leave off if the local Python trust store is broken",
    )
    args = parser.parse_args()

    parquet_path = Path(args.ssodnet_parquet)
    manifest_path = Path(args.output_manifest)
    summary_path = Path(args.output_summary)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    ssodnet_df = _load_ssodnet_main_belt(parquet_path=parquet_path)
    ssodnet_mass_selected = 0
    for row in ssodnet_df.to_dict(orient="records"):
        mass_info = _effective_mass_info(int(row["sso_number"]), row)
        if mass_info is not None and float(mass_info["mass_kg"]) > float(args.mass_min):
            ssodnet_mass_selected += 1
    jpl_rows, jpl_count = _query_jpl_candidates(
        a_min=float(args.a_min),
        a_max=float(args.a_max),
        q_min=float(args.q_min),
        data_arc_min=int(args.data_arc_min),
        n_obs_min=int(args.n_obs_min),
        limit=int(args.jpl_limit),
        verify_ssl=bool(args.verify_ssl),
    )
    selected_rows, manual_added = _build_candidate_rows(
        ssodnet_df=ssodnet_df,
        mass_min=float(args.mass_min),
        jpl_rows=jpl_rows,
        manual_include_numbers=_parse_manual_numbers(args.manual_include_numbers),
        a_min=float(args.a_min),
        a_max=float(args.a_max),
        q_min=float(args.q_min),
        data_arc_min=int(args.data_arc_min),
        n_obs_min=int(args.n_obs_min),
        verify_ssl=bool(args.verify_ssl),
    )

    manifest_rows = [
        {
            "orbit_id": str(row["orbit_id"]),
            "body_name": str(row["body_name"]),
            "split": str(row["split"]),
            "horizons_command": str(row["horizons_command"]),
            "sso_number": int(row["sso_number"]),
            "selection_source": str(row["selection_source"]),
            "mass_kg": float(row["mass_kg"]),
            "mass_source": str(row.get("mass_source", "unknown")),
            "raw_ssodnet_mass_kg": row.get("raw_ssodnet_mass_kg", None),
            "selection_note": row.get("selection_note", None),
            "semi_major_axis_au": float(row["semi_major_axis_au"]),
            "periapsis_distance_au": float(row["periapsis_distance_au"]),
            "apoapsis_distance_au": float(row["apoapsis_distance_au"]),
            "jpl_data_arc_days": int(row["jpl_data_arc_days"]),
            "jpl_n_obs_used": int(row["jpl_n_obs_used"]),
        }
        for row in selected_rows
    ]
    summary = {
        "selector": "jpl_numeric_main_belt_crossmatch_with_ssodnet_mass",
        "criteria": {
            "jpl": {
                "numbered_only": True,
                "a_min_au": float(args.a_min),
                "a_max_au": float(args.a_max),
                "q_min_au_strict": float(args.q_min),
                "data_arc_min_days": int(args.data_arc_min),
                "n_obs_min": int(args.n_obs_min),
            },
            "ssodnet": {
                "main_belt_classes": list(SSODNET_MAIN_BELT_CLASSES),
                "mass_min_kg": float(args.mass_min),
            },
            "manual_include_numbers": _parse_manual_numbers(args.manual_include_numbers),
        },
        "counts": {
            "ssodnet_mass_selected": int(ssodnet_mass_selected),
            "jpl_bulk_query_count": int(jpl_count),
            "selected_count": int(len(selected_rows)),
            "manual_added_count": int(len(manual_added)),
        },
        "manual_object_overrides": MANUAL_OBJECT_OVERRIDES,
        "manual_added": manual_added,
        "rows": selected_rows,
    }
    manifest_path.write_text(json.dumps(manifest_rows, indent=2), encoding="utf-8")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"SsODNet mass-selected subset : {ssodnet_mass_selected}")
    print(f"JPL numeric-belt candidates  : {jpl_count}")
    print(f"Selected candidates          : {len(selected_rows)}")
    if manual_added:
        print(
            "Manual add-backs             : "
            + ", ".join(f"{row['sso_number']} {row['body_name']}" for row in manual_added)
        )
    print(f"Manifest                     : {manifest_path}")
    print(f"Summary                      : {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
