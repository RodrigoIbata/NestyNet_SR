#!/usr/bin/env python3
# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata

"""Emit a units-only "dimension manifest" from an equations.txt answer key.

``equations.txt`` intermixes the ground-truth formula (the ``eqn`` column) with
the dimensional metadata (``y_units``, ``x_units``).  For blinded benchmark runs
the search must never be handed a file that contains the answer, so this script
writes a manifest that keeps only the measurement metadata and drops the
formula:

    # id vars xmin xmax y_units x_units      (eqn column removed)

The stripping is purely structural: sympy formulas contain ``()`` but never
``[]``, so the last two bracketed groups on each line are always the unit
vectors and the third-from-last is ``xmax``; everything between ``xmax`` and the
units (i.e. the formula) is discarded.  The resulting manifest is parsed by the
same ``_load_units_from_equations`` loader the pipeline already uses, since that
loader reads only the id token and the last two bracketed groups.

Usage:
    python scripts/make_units_manifest.py data/equations.txt data/equations_manifest.txt
"""

from __future__ import annotations

import argparse
import sys

from nestynet_sr.run_sr_input_utils import _extract_last_bracketed


def strip_eqn(line: str) -> str:
    """Return ``line`` with the formula (``eqn``) column removed.

    Raises ValueError if the line does not have the expected bracket structure.
    """
    x_units, rem1 = _extract_last_bracketed(line)  # rem1 = "... eqn [y_units]"
    y_units, rem2 = _extract_last_bracketed(rem1)  # rem2 = "id [vars] [xmin] [xmax] eqn"
    # xmax is the last bracketed group of rem2; the formula sits *after* it and
    # is dropped (it is the text following xmax's closing bracket, which
    # _extract_last_bracketed does not return).
    xmax, head = _extract_last_bracketed(rem2)  # head = "id [vars] [xmin]"
    return f"{head} {xmax}     {y_units}     {x_units}"


def build_manifest(src_path: str, dst_path: str) -> int:
    """Write the units manifest; return the number of data rows written."""
    n = 0
    with open(src_path, "r") as fin, open(dst_path, "w") as fout:
        fout.write("# id vars xmin xmax y_units x_units  (formula column removed; measurement metadata only)\n")
        for line in fin:
            stripped = line.strip()
            if (not stripped) or stripped.startswith("#"):
                continue
            fout.write(strip_eqn(stripped) + "\n")
            n += 1
    return n


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("src", help="Path to equations.txt (answer key + units)")
    ap.add_argument("dst", help="Output path for the units-only manifest")
    args = ap.parse_args(argv)
    n = build_manifest(args.src, args.dst)
    print(f"Wrote units manifest for {n} problems to {args.dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
