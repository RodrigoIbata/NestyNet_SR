# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata

from pathlib import Path

import numpy as np

from nestynet_sr.sr_search.config import DataHyperparams
from nestynet_sr.sr_search.data_utils import build_datasets


def test_build_datasets_interleaved_split_covers_domain(tmp_path: Path):
    csv_path = tmp_path / "toy.csv"
    rows = ["y,x0"]
    rows.extend(f"{float(i)},{float(i)}" for i in range(10))
    csv_path.write_text("\n".join(rows) + "\n", encoding="utf-8")

    data_hp = DataHyperparams(batch_size=1, ndata_select=4, ndata_select_val=3)
    data_hp.data_split_strategy = "interleaved"

    dataset_train, dataset_val, _train_loader, _val_loader = build_datasets(
        csv_path,
        Nxvars=1,
        np_dtype=np.float64,
        data_hp=data_hp,
        y_op=None,
    )

    assert dataset_train.x_arr[:, 0].tolist() == [0.0, 3.0, 6.0, 9.0]
    assert dataset_val.x_arr[:, 0].tolist() == [1.0, 4.0, 7.0]


def test_build_datasets_default_split_remains_contiguous(tmp_path: Path):
    csv_path = tmp_path / "toy.csv"
    rows = ["y,x0"]
    rows.extend(f"{float(i)},{float(i)}" for i in range(10))
    csv_path.write_text("\n".join(rows) + "\n", encoding="utf-8")

    data_hp = DataHyperparams(batch_size=1, ndata_select=4, ndata_select_val=3)

    dataset_train, dataset_val, _train_loader, _val_loader = build_datasets(
        csv_path,
        Nxvars=1,
        np_dtype=np.float64,
        data_hp=data_hp,
        y_op=None,
    )

    assert dataset_train.x_arr[:, 0].tolist() == [0.0, 1.0, 2.0, 3.0]
    assert dataset_val.x_arr[:, 0].tolist() == [4.0, 5.0, 6.0]
