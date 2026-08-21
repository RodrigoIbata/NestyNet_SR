# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

pytest.importorskip("sympy")
pytest.importorskip("scipy")

REPO_ROOT = Path(__file__).resolve().parent.parent
FEYNMAN_DE_DIR = REPO_ROOT / "examples" / "feynman_de"
if str(FEYNMAN_DE_DIR) not in sys.path:
    sys.path.insert(0, str(FEYNMAN_DE_DIR))

from oracle_de_spec_writer import write_oracle_de_spec  # noqa: E402

import nestynet_sr.sr_search.factorized_search.oracle_lab_de as oracle_lab_de_mod  # noqa: E402
from nestynet_sr.sr_search.factorized_search.oracle_lab_de import load_de_equation_spec, run_oracle_de_equation  # noqa: E402
from nestynet_sr.sr_search.config import FactorizedSearchConfig  # noqa: E402


def _write_csv(path: Path, x: np.ndarray, u: np.ndarray, du: np.ndarray, d2u: np.ndarray) -> None:
    arr = np.column_stack([u, x, du, d2u])
    np.savetxt(str(path), arr, delimiter=",", header="y,x0,du,d2u", comments="")


def test_write_oracle_de_spec_holdout_relative_paths(tmp_path: Path):
    data_dir = tmp_path / "data"
    spec_dir = tmp_path / "specs"
    data_dir.mkdir(parents=True, exist_ok=True)
    spec_dir.mkdir(parents=True, exist_ok=True)

    csvs = []
    for i in range(4):
        p = data_dir / f"de000_ic{i}.csv"
        p.write_text("y,x0\n1.0,0.0\n", encoding="utf-8")
        csvs.append((f"ic{i}", p))

    spec_path = spec_dir / "de000.spec.json"
    payload = write_oracle_de_spec(
        spec_path,
        spec_id="de000",
        trajectories=csvs,
        holdout_last_k=2,
        x_axis=0,
        order_candidates=(1, 2),
        include_x=True,
        traj_metric="max",
    )

    assert payload["split_mode"] == "traj_holdout"
    assert payload["holdout_last_k"] == 2
    assert [r["id"] for r in payload["fit_trajectories"]] == ["ic0", "ic1"]
    assert [r["id"] for r in payload["probe_trajectories"]] == ["ic2", "ic3"]
    for row in payload["fit_trajectories"] + payload["probe_trajectories"]:
        assert not Path(row["csv"]).is_absolute()

    disk_payload = json.loads(spec_path.read_text(encoding="utf-8"))
    assert disk_payload == payload


def test_write_oracle_de_spec_preserves_extra_payload(tmp_path: Path):
    data_dir = tmp_path / "data"
    spec_dir = tmp_path / "specs"
    data_dir.mkdir(parents=True, exist_ok=True)
    spec_dir.mkdir(parents=True, exist_ok=True)

    csv_path = data_dir / "de000_ic0.csv"
    csv_path.write_text("y,x0\n1.0,0.0\n", encoding="utf-8")

    spec_path = spec_dir / "de000.spec.json"
    payload = write_oracle_de_spec(
        spec_path,
        spec_id="de000",
        trajectories=[("ic0", csv_path)],
        holdout_last_k=0,
        extra={"order_preference_factor": 1.5, "note": "unit-test"},
    )

    spec = load_de_equation_spec(spec_path)
    assert payload["extra"]["order_preference_factor"] == 1.5
    assert spec.extra == {"order_preference_factor": 1.5, "note": "unit-test"}


def test_oracle_lab_de_loads_fit_probe_spec_and_probe_empty_falls_back(monkeypatch, tmp_path: Path):
    x = np.linspace(0.01, 1.0, 64, dtype=np.float64)
    u0 = np.exp(-0.4 * x)
    du0 = -0.4 * u0
    d2u0 = 0.16 * u0
    u1 = np.exp(-0.7 * x)
    du1 = -0.7 * u1
    d2u1 = 0.49 * u1

    data_dir = tmp_path / "data"
    spec_dir = tmp_path / "specs"
    data_dir.mkdir(parents=True, exist_ok=True)
    spec_dir.mkdir(parents=True, exist_ok=True)
    p0 = data_dir / "de000_ic0.csv"
    p1 = data_dir / "de000_ic1.csv"
    _write_csv(p0, x, u0, du0, d2u0)
    _write_csv(p1, x, u1, du1, d2u1)

    spec_path = spec_dir / "de000.spec.json"
    write_oracle_de_spec(
        spec_path,
        spec_id="de000",
        trajectories=[("ic0", p0), ("ic1", p1)],
        holdout_last_k=0,  # explicit fit set + empty probe set
        x_axis=0,
        order_candidates=(1,),
        include_x=True,
        traj_metric="max",
    )

    spec = load_de_equation_spec(spec_path)
    assert spec.split_mode == "per_traj_point"
    assert len(spec.fit_trajectories) == 2
    assert len(spec.probe_trajectories) == 0
    for tr in spec.fit_trajectories:
        assert Path(tr.csv).is_absolute()

    class _FakeArch:
        def best(self, k):
            rec = SimpleNamespace(
                best_mse=1.0e-9,
                best_expr=("var", 0),
                mapping={"kind": "poly", "coeffs": [0.0, 1.0], "mu": 0.0, "std": 1.0},
            )
            return [rec]

    monkeypatch.setattr(oracle_lab_de_mod, "run_explorer_core", lambda **kwargs: _FakeArch())

    hp = FactorizedSearchConfig()
    hp.n_iter = 80
    hp.max_depth = 3
    hp.poly_degree = 3
    hp.return_topk = 1
    hp.n_fit = 48
    hp.n_probe = 48
    hp.n_seeds = 1
    hp.split_iter_across_seeds = True
    hp.brute_depth = 1
    hp.refine_enable = False

    report = run_oracle_de_equation(spec, factorized_search_hp=hp, seed=13, dtype=torch.float64, verbose=False)
    assert report["best"] is not None
    assert report["split_mode"] == "per_traj_point"
    assert len(report["fit_trajectories"]) == 2
    assert len(report["probe_trajectories"]) == 0

    order_row = report["per_order"][0]
    assert int(order_row["n_traj_fit"]) == 2
    assert int(order_row["n_traj_probe"]) == 2
