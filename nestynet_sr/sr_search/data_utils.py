# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""
Helpers to construct train/validation datasets and dataloaders.
"""

import nestynet
import numpy as np
import torch


def _slice_rows(obj, start: int, stop: int):
    if obj is None:
        return None
    if hasattr(obj, "iloc"):
        return obj.iloc[start:stop].reset_index(drop=True)
    return obj[start:stop]


def _take_rows(obj, indices):
    if obj is None:
        return None
    if hasattr(obj, "iloc"):
        return obj.iloc[list(indices)].reset_index(drop=True)
    return obj[list(indices)]


def _evenly_spaced_indices(n_total: int, n_take: int) -> np.ndarray:
    n_total_i = int(n_total)
    n_take_i = int(n_take)
    if n_take_i <= 0:
        return np.empty(0, dtype=np.int64)
    if n_take_i >= n_total_i:
        return np.arange(n_total_i, dtype=np.int64)
    return np.floor(np.linspace(0, n_total_i - 1, n_take_i)).astype(np.int64)


def _interleaved_row_order(n_total: int, n_train: int, n_val: int) -> list[int]:
    n_total_i = int(n_total)
    n_train_i = int(n_train)
    n_val_i = int(n_val)
    n_selected = n_train_i + n_val_i
    if n_total_i <= 0 or n_selected <= 0 or n_selected > n_total_i:
        return list(range(n_total_i))

    selected = _evenly_spaced_indices(n_total_i, n_selected)
    if n_val_i <= 0:
        val_pos = np.empty(0, dtype=np.int64)
    else:
        val_pos = np.floor((np.arange(n_val_i, dtype=np.float64) + 0.5) * n_selected / n_val_i)
        val_pos = np.clip(val_pos.astype(np.int64), 0, n_selected - 1)
        val_pos = np.unique(val_pos)
        if int(val_pos.size) < n_val_i:
            missing = [p for p in range(n_selected) if p not in set(val_pos.tolist())]
            val_pos = np.array(sorted(val_pos.tolist() + missing[: n_val_i - int(val_pos.size)]), dtype=np.int64)

    is_val = np.zeros(n_selected, dtype=bool)
    is_val[val_pos[:n_val_i]] = True
    train_idx = selected[~is_val]
    val_idx = selected[is_val]
    selected_set = set(int(i) for i in selected.tolist())
    remainder = [i for i in range(n_total_i) if i not in selected_set]
    return [int(i) for i in train_idx.tolist()] + [int(i) for i in val_idx.tolist()] + remainder


def _reorder_loaded_rows(out, order):
    if len(out) == 3:
        df, df_ref, nx = out
        return _take_rows(df, order), _take_rows(df_ref, order), nx
    if len(out) == 4:
        df, df_ref, nx, df_sigma = out
        return (
            _take_rows(df, order),
            _take_rows(df_ref, order),
            nx,
            _take_rows(df_sigma, order),
        )
    raise ValueError("data_loader must return (X_df,Y_df,nx) or (X_df,Y_df,nx,Y_sigma_df)")


def _split_csv_loader(
    filepath,
    *,
    _data_slice_start: int | None,
    _data_slice_stop: int | None,
    _data_split_strategy: str,
    _data_split_n_train: int,
    _data_split_n_val: int,
    **kwargs,
):
    out = nestynet.dataloader.get_csv_data_as_pandas(filepath, **kwargs)
    if len(out) == 3:
        df, df_ref, nx = out
        out = (
            _slice_rows(df, int(_data_slice_start or 0), _data_slice_stop),
            _slice_rows(df_ref, int(_data_slice_start or 0), _data_slice_stop),
            nx,
        )
    elif len(out) == 4:
        df, df_ref, nx, df_sigma = out
        out = (
            _slice_rows(df, int(_data_slice_start or 0), _data_slice_stop),
            _slice_rows(df_ref, int(_data_slice_start or 0), _data_slice_stop),
            nx,
            _slice_rows(df_sigma, int(_data_slice_start or 0), _data_slice_stop),
        )
    else:
        raise ValueError("data_loader must return (X_df,Y_df,nx) or (X_df,Y_df,nx,Y_sigma_df)")
    if str(_data_split_strategy).strip().lower() == "interleaved":
        n_total = len(out[0])
        order = _interleaved_row_order(
            n_total,
            int(_data_split_n_train),
            int(_data_split_n_val),
        )
        out = _reorder_loaded_rows(out, order)
    return out


def _loader_and_kwargs_for_data_split(data_hp):
    data_slice = int(getattr(data_hp, "data_slice", 0) or 0)
    if data_slice < 0:
        raise ValueError(f"data_slice must be >= 0, got {data_slice}")
    split_strategy = str(getattr(data_hp, "data_split_strategy", "contiguous") or "contiguous").strip().lower()
    if split_strategy not in ("contiguous", "interleaved"):
        raise ValueError(f"Unsupported data_split_strategy={split_strategy!r}")
    if data_slice == 0 and split_strategy == "contiguous":
        return nestynet.dataloader.get_csv_data_as_pandas, None
    n_train = int(getattr(data_hp, "ndata_select", 0) or 0)
    n_val = int(getattr(data_hp, "ndata_select_val", 0) or 0)
    if n_train <= 0 or n_val <= 0:
        raise ValueError(
            "data_slice requires positive ndata_select and ndata_select_val "
            f"(got {n_train}, {n_val})"
        )
    block = n_train + n_val
    start = int(data_slice * block) if data_slice > 0 else None
    stop = int(start + block) if start is not None else None
    return _split_csv_loader, {
        "_data_slice_start": start,
        "_data_slice_stop": stop,
        "_data_split_strategy": split_strategy,
        "_data_split_n_train": n_train,
        "_data_split_n_val": n_val,
    }


def build_datasets(filepath, Nxvars, np_dtype, data_hp, y_op):
    """
    Build training and validation datasets + non-shuffled dataloaders.

    Note: If there is overlap between training and validation datasets, the LM optimizer
    will raise a stern warning, allowing users to adjust parameters as needed.

    Returns
    -------
    dataset_train, dataset_val, train_loader, val_loader
    or (None, None, None, None) if y_op caused an error.
    """
    data_loader, data_function_kwargs = _loader_and_kwargs_for_data_split(data_hp)

    dataset_train = nestynet.dataloader.PhysDataset(
        filepath,
        mode="train",
        # NestyNet's default split_policy="random" would permute train/validation
        # membership. Row order here already encodes data_split_strategy
        # (contiguous or interleaved), so the split must follow source order.
        split_policy="contiguous",
        data_loader=data_loader,
        ndata_select=data_hp.ndata_select,
        ndata_select_val=data_hp.ndata_select_val,
        Nxvars=Nxvars,
        np_dtype=np_dtype,
        y_operator=y_op,
        data_function_kwargs=data_function_kwargs,
    )
    if getattr(dataset_train, "y_op_error", False):
        y_op_str = y_op.__name__ if hasattr(y_op, "__name__") else str(y_op)
        print("Skipping {} due to error in y_operator (train)".format(y_op_str))
        return None, None, None, None

    dataset_val = nestynet.dataloader.PhysDataset(
        filepath,
        mode="validation",
        split_policy="contiguous",  # must match dataset_train; see note above
        data_loader=data_loader,
        ndata_select=data_hp.ndata_select,
        ndata_select_val=data_hp.ndata_select_val,
        Nxvars=Nxvars,
        np_dtype=np_dtype,
        y_operator=y_op,
        data_function_kwargs=data_function_kwargs,
    )
    if getattr(dataset_val, "y_op_error", False):
        y_op_str = y_op.__name__ if hasattr(y_op, "__name__") else str(y_op)
        print("Skipping {} due to error in y_operator (validation)".format(y_op_str))
        return None, None, None, None

    # Validate that datasets are large enough for the requested batch size
    if len(dataset_train) < data_hp.batch_size:
        raise ValueError(
            f"Training dataset has {len(dataset_train)} samples but batch size is {data_hp.batch_size}. "
            f"Reduce batch_size in DataHyperparams or provide more training data with ndata_select."
        )
    if len(dataset_val) < data_hp.batch_size:
        raise ValueError(
            f"Validation dataset has {len(dataset_val)} samples but batch size is {data_hp.batch_size}. "
            f"Reduce batch_size in DataHyperparams or provide more validation data with ndata_select_val."
        )

    train_loader = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=data_hp.batch_size,
        shuffle=False,
        drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=data_hp.batch_size,
        shuffle=False,
        drop_last=True,
    )

    print(
        "\n\nSize of datasets: training: {}, val: {}".format(len(dataset_train), len(dataset_val))
    )

    return dataset_train, dataset_val, train_loader, val_loader


def build_datasets_multi(filepaths, Nxvars, np_dtype, data_hp, y_op):
    """Build training/validation datasets + loaders for multiple CSVs.

    This is a thin wrapper around :func:`build_datasets` that returns lists.

    Notes
    -----
    * For a given y-transform, we require that *all* datasets can be built.
      If any dataset fails due to y_op domain errors, this returns
      (None, None, None, None) to signal "skip this y_op".
    * Nxvars is assumed to be consistent across all CSVs.
    """
    if filepaths is None:
        raise ValueError("filepaths must be a non-empty sequence")
    if isinstance(filepaths, (str, bytes)):
        filepaths = [str(filepaths)]
    filepaths = [str(p) for p in filepaths]
    if len(filepaths) == 0:
        raise ValueError("filepaths must be a non-empty sequence")

    ds_tr_list = []
    ds_va_list = []
    dl_tr_list = []
    dl_va_list = []

    for fp in filepaths:
        ds_tr, ds_va, dl_tr, dl_va = build_datasets(fp, Nxvars, np_dtype, data_hp, y_op)
        if dl_tr is None or dl_va is None:
            return None, None, None, None
        ds_tr_list.append(ds_tr)
        ds_va_list.append(ds_va)
        dl_tr_list.append(dl_tr)
        dl_va_list.append(dl_va)

    return ds_tr_list, ds_va_list, dl_tr_list, dl_va_list
