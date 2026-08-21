# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata

from pathlib import Path
import json

import pytest
import torch

pytest.importorskip("sympy")

from nestynet_sr.sr_search.factorized_search.oracle_pretrain import run_oracle_pretrain_pipeline
from nestynet_sr.sr_search.config import FactorizedSearchConfig


def _payload() -> dict:
    return {
        "id": "oracle_pretrain_runner_demo",
        "basis": ["L"],
        "variables": [
            {"name": "x", "bounds": [0.3, 3.2], "dim": [1]},
        ],
        "constants": [],
        "target": {
            "expr": "sin(x) + x*(x + x)",
            "dim": [1],
        },
    }


def test_run_oracle_pretrain_pipeline_smoke(tmp_path: Path):
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(_payload(), indent=2), encoding="utf-8")

    hp = FactorizedSearchConfig()
    hp.n_fit = 32
    hp.n_probe = 48
    hp.poly_degree = 3
    hp.inverse_max_paths = 4
    hp.inverse_topk_terms = 3
    hp.inverse_shortlist_mult = 2

    out = run_oracle_pretrain_pipeline(
        [spec_path],
        output_dir=tmp_path / "out",
        factorized_search_hp=hp,
        seeds=[0, 1, 2, 3],
        dtype=torch.float64,
        depth_min=2,
        depth_max=8,
        topk=4,
        max_corrupt_paths_per_spec=1,
        sweep_max_paths=4,
        hidden_dim=16,
        pretrain_epochs=20,
        pretrain_lr=2.0e-2,
        verbose=False,
    )

    assert int(out["n_curriculum_rows"]) >= 1
    assert Path(out["dataset_path"]).exists()
    assert Path(out["pretrain_bundle_path"]).exists()
    assert Path(out["final_bundle_path"]).exists()
