#!/usr/bin/env python3
# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata

from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path
import re
import ssl
from typing import Any
import urllib.parse
import urllib.request

from astropy import units as u
from astropy.time import Time

AU_KM = float((1.0 * u.au).to_value(u.km))
SECONDS_PER_DAY = 86400.0
HORIZONS_API_URL = "https://ssd.jpl.nasa.gov/api/horizons.api"


def _stop_date_from_years(start_date: str, years: float) -> str:
    stop = Time(str(start_date), scale="tdb") + float(years) * 365.25 * u.day
    return str(stop.utc.isot[:10])


def _request_horizons_text(
    *,
    command: str,
    start_date: str,
    stop_date: str,
    cadence_days: float,
    verify_ssl: bool,
) -> str:
    params = {
        "format": "text",
        "COMMAND": f"'{str(command)}'",
        "OBJ_DATA": "NO",
        "MAKE_EPHEM": "YES",
        "EPHEM_TYPE": "VECTORS",
        "CENTER": "'500@10'",
        "START_TIME": f"'{str(start_date)}'",
        "STOP_TIME": f"'{str(stop_date)}'",
        "STEP_SIZE": f"'{float(cadence_days):g} d'",
        "CSV_FORMAT": "YES",
        "OUT_UNITS": "KM-S",
        "VEC_TABLE": "2",
        "VEC_CORR": "'NONE'",
    }
    url = HORIZONS_API_URL + "?" + urllib.parse.urlencode(params)
    context = None if bool(verify_ssl) else ssl._create_unverified_context()
    with urllib.request.urlopen(url, timeout=120, context=context) as response:
        return response.read().decode("utf-8")


def parse_horizons_vectors_text(text: str) -> dict[str, Any]:
    lines = str(text).splitlines()
    if "$$SOE" not in text or "$$EOE" not in text:
        raise ValueError("HORIZONS response did not contain a $$SOE/$$EOE vector block")

    header = {}
    target_name = None
    center_name = None
    for line in lines:
        if line.startswith("Target body name:"):
            target_name = line.split(":", 1)[1].strip()
        elif line.startswith("Center body name:"):
            center_name = line.split(":", 1)[1].strip()
        elif line.startswith("Start time"):
            header["start_time"] = line.split(":", 1)[1].strip()
        elif line.startswith("Stop  time"):
            header["stop_time"] = line.split(":", 1)[1].strip()
        elif line.startswith("Step-size"):
            header["step_size"] = line.split(":", 1)[1].strip()

    soe_idx = lines.index("$$SOE")
    eoe_idx = lines.index("$$EOE")
    data_lines = [line.strip() for line in lines[soe_idx + 1 : eoe_idx] if line.strip()]
    rows = []
    for raw_line in data_lines:
        reader = csv.reader(io.StringIO(raw_line), skipinitialspace=True)
        fields = next(reader)
        if len(fields) < 8:
            raise ValueError(f"expected at least 8 CSV fields in vector row, got {fields!r}")
        jd_tdb = float(fields[0])
        calendar_date = str(fields[1]).strip()
        x_km = float(fields[2])
        y_km = float(fields[3])
        z_km = float(fields[4])
        vx_km_s = float(fields[5])
        vy_km_s = float(fields[6])
        vz_km_s = float(fields[7])
        rows.append(
            {
                "jd": jd_tdb,
                "calendar_date_tdb": calendar_date,
                "x_au": x_km / AU_KM,
                "y_au": y_km / AU_KM,
                "z_au": z_km / AU_KM,
                "vx_au_per_d": vx_km_s * SECONDS_PER_DAY / AU_KM,
                "vy_au_per_d": vy_km_s * SECONDS_PER_DAY / AU_KM,
                "vz_au_per_d": vz_km_s * SECONDS_PER_DAY / AU_KM,
            }
        )
    if not rows:
        raise ValueError("parsed zero state rows from HORIZONS response")

    jd0 = float(rows[0]["jd"])
    for row in rows:
        row["t_day"] = float(row["jd"] - jd0)

    return {
        "target_name": target_name,
        "center_name": center_name,
        "header": header,
        "rows": rows,
    }


def _sanitize_stem(text: str) -> str:
    out = re.sub(r"[^A-Za-z0-9._-]+", "_", str(text).strip().lower())
    return out.strip("_") or "body"


def _write_state_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "t_day",
        "jd",
        "calendar_date_tdb",
        "x_au",
        "y_au",
        "z_au",
        "vx_au_per_d",
        "vy_au_per_d",
        "vz_au_per_d",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fetch heliocentric state vectors from NASA JPL HORIZONS and normalize them for the ephemeris Kepler scaffold",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--source_manifest",
        type=str,
        default=str(Path(__file__).resolve().parent / "data" / "horizons_curated_sources.json"),
        help="JSON list of bodies with orbit_id/body_name/horizons_command/split",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(Path(__file__).resolve().parent / "data" / "raw"),
        help="Directory where normalized state CSVs will be written",
    )
    parser.add_argument(
        "--output_manifest",
        type=str,
        default=str(Path(__file__).resolve().parent / "data" / "raw_states_manifest.json"),
        help="JSON manifest consumed by --provider raw_csv",
    )
    parser.add_argument("--start_date", type=str, default="1980-01-01", help="HORIZONS start date")
    parser.add_argument("--years", type=float, default=30.0, help="Trajectory span in years")
    parser.add_argument("--cadence_days", type=float, default=1.0, help="Cadence in days")
    parser.add_argument(
        "--verify_ssl",
        action="store_true",
        help="Verify HTTPS certificates; leave off if the local Python trust store is broken",
    )
    args = parser.parse_args()

    source_manifest_path = Path(args.source_manifest)
    source_rows = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    output_dir = Path(args.output_dir)
    output_manifest_path = Path(args.output_manifest)
    output_manifest_path.parent.mkdir(parents=True, exist_ok=True)

    stop_date = _stop_date_from_years(str(args.start_date), float(args.years))
    normalized_rows = []
    for row in list(source_rows):
        orbit_id = str(row["orbit_id"])
        body_name = str(row.get("body_name", orbit_id))
        horizons_command = str(row.get("horizons_command", orbit_id))
        split = str(row["split"])
        text = _request_horizons_text(
            command=horizons_command,
            start_date=str(args.start_date),
            stop_date=stop_date,
            cadence_days=float(args.cadence_days),
            verify_ssl=bool(args.verify_ssl),
        )
        parsed = parse_horizons_vectors_text(text)
        csv_path = output_dir / f"{_sanitize_stem(orbit_id)}.csv"
        _write_state_csv(csv_path, list(parsed["rows"]))
        normalized_rows.append(
            {
                "orbit_id": orbit_id,
                "body_name": body_name,
                "split": split,
                "csv_path": str(csv_path),
                "horizons_command": horizons_command,
                "target_name": parsed.get("target_name", None),
                "center_name": parsed.get("center_name", None),
                "start_date": str(args.start_date),
                "stop_date": stop_date,
                "cadence_days": float(args.cadence_days),
                "n_rows": int(len(parsed["rows"])),
            }
        )
        print(
            f"{orbit_id}: fetched {len(parsed['rows'])} rows from HORIZONS "
            f"({parsed.get('target_name', body_name)})"
        )

    output_manifest_path.write_text(json.dumps(normalized_rows, indent=2), encoding="utf-8")
    print(f"Raw state manifest: {output_manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
