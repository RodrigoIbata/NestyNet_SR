# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata

"""Build the SPARC baryonic-acceleration-carrier pilot dataset.

Parses the SPARC (Lelli, McGaugh & Schombert 2016) machine-readable tables in
``data/sparc/`` and emits per-row component accelerations grouped by galaxy:

    g_gas  = Vgas |Vgas| / R      (signed; Vgas < 0 marks central HI depressions)
    g_disk = Vdisk|Vdisk| / R     (M/L = 1 at 3.6um, so the carrier coefficient
                                   is the disk mass-to-light ratio Upsilon_d)
    g_bul  = Vbul |Vbul| / R      (M/L = 1; zero for bulgeless galaxies)
    g_obs  = Vobs^2 / R

Row-quality cuts follow the standard SPARC RAR analysis (Lelli et al. 2017):
quality flag Q < 3, inclination >= 30 deg, e_Vobs/Vobs < 0.1, Vobs > 0.

Outputs (in ``--outdir``):
    sparc_carrier_bulgeless.csv   galaxies with Vbul == 0 at every radius
    sparc_carrier_all.csv         every surviving galaxy (bulge column included)
    sparc_carrier_galaxies.csv    one row per surviving galaxy (metadata)
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

KPC_M = 3.0856775814913673e19  # kpc in metres
KMS2_PER_KPC_TO_SI = 1.0e6 / KPC_M  # (km/s)^2 / kpc -> m/s^2

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = REPO_ROOT / "data" / "sparc"


def _f(s: str) -> float:
    s = s.strip()
    return float(s) if s else math.nan


def _data_lines(path: Path) -> list[str]:
    """Lines after the last '----' separator of an MRT file."""
    with open(path) as fh:
        lines = fh.readlines()
    last_sep = max(i for i, ln in enumerate(lines) if ln.startswith("----"))
    return [ln for ln in lines[last_sep + 1:] if ln.strip()]


def parse_table1(path: Path) -> dict[str, dict]:
    """Galaxy-level properties, keyed by galaxy name.

    Whitespace-split: SPARC names contain no spaces and every numeric field up
    to Q is always populated (the file's real alignment drifts from the byte
    header, so fixed-width slicing is unsafe here).
    """
    galaxies: dict[str, dict] = {}
    for ln in _data_lines(path):
        t = ln.split()
        galaxies[t[0]] = {
            "T": int(t[1]),
            "D": _f(t[2]),
            "e_D": _f(t[3]),
            "Inc": _f(t[5]),
            "e_Inc": _f(t[6]),
            "L36": _f(t[7]),
            "MHI": _f(t[13]),
            "Vflat": _f(t[15]),
            "Q": int(t[17]),
        }
    return galaxies


def parse_table2(path: Path) -> list[dict]:
    """Per-radius mass-model rows (whitespace-split; all 10 columns always present)."""
    rows: list[dict] = []
    for ln in _data_lines(path):
        t = ln.split()
        rows.append({
            "galaxy": t[0],
            "D": _f(t[1]),
            "R": _f(t[2]),
            "Vobs": _f(t[3]),
            "e_Vobs": _f(t[4]),
            "Vgas": _f(t[5]),
            "Vdisk": _f(t[6]),
            "Vbul": _f(t[7]),
        })
    return rows


def signed_acc(v: float, r: float) -> float:
    """v|v|/R in m/s^2 with V in km/s and R in kpc."""
    return v * abs(v) / r * KMS2_PER_KPC_TO_SI


def build(data_dir: Path, outdir: Path, min_rows: int = 5) -> None:
    gal = parse_table1(data_dir / "SPARC_Lelli2016c.mrt")
    rows = parse_table2(data_dir / "MassModels_Lelli2016c.mrt")

    by_gal: dict[str, list[dict]] = defaultdict(list)
    n_row_cut = 0
    for r in rows:
        g = gal.get(r["galaxy"])
        if g is None:
            continue
        # Galaxy-level cuts: quality and inclination (standard RAR cuts).
        if g["Q"] >= 3 or not (g["Inc"] >= 30.0):
            continue
        # Row-level cuts.
        if not (r["R"] > 0 and r["Vobs"] > 0):
            n_row_cut += 1
            continue
        if not (r["e_Vobs"] / r["Vobs"] < 0.1):
            n_row_cut += 1
            continue
        by_gal[r["galaxy"]].append(r)

    kept = {name: rs for name, rs in by_gal.items() if len(rs) >= min_rows}

    outdir.mkdir(parents=True, exist_ok=True)
    cols = ["galaxy", "R_kpc", "g_gas", "g_disk", "g_bul", "g_obs",
            "e_gobs_frac", "Vobs", "e_Vobs", "Vgas", "Vdisk", "Vbul"]

    def emit(path: Path, names: list[str], include_bulge: bool) -> int:
        n = 0
        with open(path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(cols if include_bulge else [c for c in cols if c != "g_bul"])
            for name in names:
                for r in kept[name]:
                    g_gas = signed_acc(r["Vgas"], r["R"])
                    g_disk = signed_acc(r["Vdisk"], r["R"])
                    g_bul = signed_acc(r["Vbul"], r["R"])
                    g_obs = r["Vobs"] ** 2 / r["R"] * KMS2_PER_KPC_TO_SI
                    e_frac = 2.0 * r["e_Vobs"] / r["Vobs"]
                    row = [name, r["R"], g_gas, g_disk, g_bul, g_obs,
                           e_frac, r["Vobs"], r["e_Vobs"], r["Vgas"], r["Vdisk"], r["Vbul"]]
                    if not include_bulge:
                        row = row[:4] + row[5:]
                    w.writerow(row)
                    n += 1
        return n

    bulgeless = sorted(n for n in kept if all(r["Vbul"] == 0.0 for r in kept[n]))
    all_names = sorted(kept)
    # "Gold" kinematically-safer subsample (preregistered cuts, per Neil's
    # revision): bulgeless AND intermediate inclination 40-75 deg (projection
    # gauge under control, not near-face-on) AND regular late-type disks
    # T in [4, 9] (Sbc..Sm, avoiding the most irregular dwarfs) AND >= 8 rows.
    gold = sorted(n for n in bulgeless
                  if 40.0 <= gal[n]["Inc"] <= 75.0
                  and 4 <= gal[n]["T"] <= 9
                  and len(kept[n]) >= 8)

    n_bl = emit(outdir / "sparc_carrier_bulgeless.csv", bulgeless, include_bulge=False)
    n_gold = emit(outdir / "sparc_carrier_gold.csv", gold, include_bulge=False)
    n_all = emit(outdir / "sparc_carrier_all.csv", all_names, include_bulge=True)

    with open(outdir / "sparc_carrier_galaxies.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["galaxy", "bulgeless", "n_rows", "T", "D_Mpc", "e_D", "Inc_deg",
                    "e_Inc", "L36_1e9Lsun", "MHI_1e9Msun", "Vflat", "Q"])
        for name in all_names:
            g = gal[name]
            w.writerow([name, int(name in set(bulgeless)), len(kept[name]), g["T"],
                        g["D"], g["e_D"], g["Inc"], g["e_Inc"], g["L36"], g["MHI"],
                        g["Vflat"], g["Q"]])

    print(f"galaxies surviving cuts : {len(all_names)} "
          f"({len(bulgeless)} bulgeless, {len(gold)} gold)")
    print(f"rows                    : {n_all} total, {n_bl} bulgeless, {n_gold} gold")
    print(f"rows cut (row-level)    : {n_row_cut}")
    print(f"wrote -> {outdir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data_dir", type=Path, default=DEFAULT_DATA_DIR)
    ap.add_argument("--outdir", type=Path, default=Path(__file__).resolve().parent / "data")
    ap.add_argument("--min_rows", type=int, default=5)
    args = ap.parse_args()
    build(args.data_dir, args.outdir, args.min_rows)
