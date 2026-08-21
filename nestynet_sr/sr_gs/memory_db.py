# SPDX-License-Identifier: MPL-2.0

"""Static paper↔code map for the NestyNet_SR_GS prototype.

The JSON file written by ``write_default_memory_db`` is meant as an internal
navigation index: it records where each theoretical construct from the papers is
implemented or extended in this repository.  It is not used as a runtime cache.
"""

from __future__ import annotations

import json
from pathlib import Path


DEFAULT_MEMORY_DB = {
    "papers": {
        "NestyNet I": {
            "concepts": {
                "segmented softplus surrogate": ["nestynet (external dependency)", "nestynet_sr/sr_search/training.py"],
                "analytic gradients/Hessians": ["nestynet_sr/sr_de/de_search.py", "nestynet_sr/sr_search/search.py"],
                "predictive LM": ["nestynet optimizer dependency", "nestynet_sr/sr_search/training.py"],
            }
        },
        "NestyNet II": {
            "concepts": {
                "Stage A separability": ["nestynet_sr/sr_search/search.py"],
                "compound coordinates": ["nestynet_sr/sr_core/separability_math.py", "nestynet_sr/sr_search/compound_proposals/"],
                "Stage B rewrites": ["nestynet_sr/sr_search/stageB/"],
                "factorized symbolic search": ["nestynet_sr/sr_search/factorized_search/"],
                "dimensional analysis": ["nestynet_sr/sr_core/units.py", "nestynet_sr/sr_core/buckingham_sudoku.py"],
                "generalized symmetries prototype": ["nestynet_sr/sr_gs/", "examples/generalized_symmetries/"],
            }
        },
        "NestyNet III": {
            "concepts": {
                "sparse DE discovery": ["nestynet_sr/sr_de/de_search.py"],
                "Basin-DE / factorized DE": ["nestynet_sr/sr_de/factorized_de.py", "nestynet_sr/sr_search/factorized_search/oracle_lab_de.py"],
                "feynman_de benchmark": ["examples/feynman_de/"],
                "GS hard-tail DE templates": ["nestynet_sr/sr_gs/de_bridge.py"],
            }
        },
    },
    "gs_ablation_switches": {
        "SR Stage A": [
            "--gs-enable",
            "--gs-mode {off,audit,propose}",
            "--gs-no-translations",
            "--gs-no-diagonal-translations",
            "--gs-no-scalings",
            "--gs-no-rotations",
            "--gs-lorentz-boosts",
            "--gs-no-output-equivariance",
        ],
        "DE": [
            "--gs-enable",
            "--de-hard-tail-templates",
            "--de-hard-tail-no-radial-templates",
            "--de-hard-tail-velocity-templates",
        ],
    },
}


def write_default_memory_db(path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(DEFAULT_MEMORY_DB, indent=2, sort_keys=True), encoding="utf-8")
    return path
