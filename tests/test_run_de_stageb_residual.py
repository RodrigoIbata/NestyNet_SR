# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
"""Tests for run_de Stage-B residual refinement helpers."""

from types import SimpleNamespace

import numpy as np
import torch

from nestynet_sr.run_de import (
    ZeroTargetDataset,
    _make_zero_target_loader,
    _run_stageb_residual_refine_multi,
    _run_stageb_residual_refine_single,
)
from nestynet_sr.sr_core.bridges import Add, DU, FreeConst
from nestynet_sr.sr_de.de_search import DESearchConfig


class _TinyDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.x = torch.tensor([[0.0], [1.0], [2.0]], dtype=torch.float64)
        self.y = torch.tensor([[1.5], [2.5], [3.5]], dtype=torch.float64)
        self.extra = torch.tensor([[10.0], [20.0], [30.0]], dtype=torch.float64)

    def __len__(self):
        return int(self.x.shape[0])

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.extra[idx]


class _TinyNumpyDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.x = np.array([[0.0], [1.0], [2.0]], dtype=np.float64)
        self.y = np.array([[1.5], [2.5], [3.5]], dtype=np.float64)

    def __len__(self):
        return int(self.x.shape[0])

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def _dummy_residual(tag: str, init: float):
    return Add(DU(0), FreeConst(name=tag, tag=tag, init=float(init)))


def test_zero_target_dataset_and_loader_preserve_shapes_and_tail_fields():
    base = _TinyDataset()
    wrapped = ZeroTargetDataset(base)

    x0, z0, e0 = wrapped[0]
    xb, yb, eb = base[0]
    assert torch.allclose(x0, xb)
    assert torch.allclose(z0, torch.zeros_like(yb))
    assert torch.allclose(e0, eb)

    ref_loader = torch.utils.data.DataLoader(base, batch_size=2, shuffle=False, drop_last=True)
    z_loader = _make_zero_target_loader(base, ref_loader, fallback_batch_size=8)
    assert int(z_loader.batch_size) == 2
    assert bool(z_loader.drop_last) is True

    _, y_batch, extra_batch = next(iter(z_loader))
    assert torch.allclose(y_batch, torch.zeros_like(y_batch))
    assert torch.allclose(extra_batch, torch.tensor([[10.0], [20.0]], dtype=torch.float64))


def test_zero_target_dataset_supports_numpy_targets():
    base = _TinyNumpyDataset()
    wrapped = ZeroTargetDataset(base)
    x0, z0 = wrapped[0]
    assert isinstance(z0, np.ndarray)
    assert np.allclose(z0, np.zeros_like(base.y[0]))

    ref_loader = torch.utils.data.DataLoader(base, batch_size=2, shuffle=False, drop_last=False)
    z_loader = _make_zero_target_loader(base, ref_loader, fallback_batch_size=4)
    _, y_batch = next(iter(z_loader))
    assert torch.is_tensor(y_batch)
    assert torch.allclose(y_batch, torch.zeros_like(y_batch))


def test_stageb_single_refine_forwards_atom_factory_and_zero_targets(monkeypatch):
    from nestynet_sr.sr_de import de_search
    from nestynet_sr.sr_core import bridges as core_bridges
    from nestynet_sr.sr_search.stageB import fitting as stageb_fitting
    from nestynet_sr.sr_search.stageB import atom_mapping as stageb_atom_mapping

    captured = {}
    atom_factory_token = object()

    class _DummyLeaf:
        def __init__(self, value: float):
            self.model = SimpleNamespace(
                value=torch.tensor(float(value), dtype=torch.float64),
            )

    def _fake_make_u_feature_atom_factory(_surrogate, *, cache=None):
        return atom_factory_token

    def _fake_fit_candidate_root(**kwargs):
        captured["atom_factory"] = kwargs.get("atom_factory")
        _, yv, _ = next(iter(kwargs["val_loader"]))
        captured["val_targets"] = yv
        return SimpleNamespace(
            val_loss=4.0,
            root=kwargs["root"],
            model=SimpleNamespace(name="dummy"),
        )

    def _fake_build_atom_to_leaf_map(root, _model):
        atoms = core_bridges.collect_all_atoms(root)
        c_atom = next(a for a in atoms if str(getattr(a, "kind", "")).lower() in ("free_const", "scale", "fixed_const"))
        return {id(c_atom): _DummyLeaf(9.0)}

    monkeypatch.setattr(de_search, "make_u_feature_atom_factory", _fake_make_u_feature_atom_factory)
    monkeypatch.setattr(stageb_fitting, "_fit_candidate_root", _fake_fit_candidate_root)
    monkeypatch.setattr(stageb_atom_mapping, "build_atom_to_leaf_map", _fake_build_atom_to_leaf_map)

    base = _TinyDataset()
    dl = torch.utils.data.DataLoader(base, batch_size=2, shuffle=False, drop_last=False)
    res = SimpleNamespace(
        residual_ast=_dummy_residual(tag="c0", init=1.0),
        order=1,
        x_axis=0,
        term_asts=[None],
        coeffs=torch.tensor([1.0], dtype=torch.float64),
        rms_train=0.0,
        rms_val=0.0,
    )

    _state, meta = _run_stageb_residual_refine_single(
        res=res,
        surrogate=object(),
        ds_tr=base,
        ds_va=base,
        dl_tr=dl,
        dl_va=dl,
        lm_hp=SimpleNamespace(),
        cfg=DESearchConfig(),
        device=torch.device("cpu"),
        dtype=torch.float64,
        epochs_stageB=7,
    )

    assert captured["atom_factory"] is atom_factory_token
    assert torch.allclose(captured["val_targets"], torch.zeros_like(captured["val_targets"]))
    assert float(meta["val_mse"]) == 4.0
    assert float(meta["val_rms"]) == 2.0
    assert int(meta["epochs"]) == 7
    assert bool(meta["coefficients_updated"]) is True
    assert [float(v) for v in res.coeffs.tolist()] == [9.0]


def test_stageb_multi_refine_forwards_per_dataset_atom_factories(monkeypatch):
    from nestynet_sr.sr_de import de_search
    from nestynet_sr.sr_core import bridges as core_bridges
    from nestynet_sr.sr_search.stageB import fitting as stageb_fitting
    from nestynet_sr.sr_search.stageB import atom_mapping as stageb_atom_mapping

    captured = {}

    class _DummyLeaf:
        def __init__(self, value: float):
            self.model = SimpleNamespace(
                value=torch.tensor(float(value), dtype=torch.float64),
            )

    def _fake_make_u_feature_atom_factory(surrogate, *, cache=None):
        return f"factory:{surrogate}"

    def _fake_fit_candidate_root_multi(**kwargs):
        captured["atom_factory"] = kwargs.get("atom_factory")
        captured["val_targets"] = [next(iter(dl))[1] for dl in kwargs["val_loaders"]]
        return SimpleNamespace(
            val_loss=1.0,
            val_losses=[1.0, 4.0],
            root=kwargs["root"],
            model=SimpleNamespace(coeff_value=1.25),
            models=[SimpleNamespace(coeff_value=1.25), SimpleNamespace(coeff_value=2.5)],
        )

    def _fake_build_atom_to_leaf_map(root, model):
        atoms = core_bridges.collect_all_atoms(root)
        c_atom = next(a for a in atoms if str(getattr(a, "kind", "")).lower() in ("free_const", "scale", "fixed_const"))
        return {id(c_atom): _DummyLeaf(float(getattr(model, "coeff_value", 1.0)))}

    monkeypatch.setattr(de_search, "make_u_feature_atom_factory", _fake_make_u_feature_atom_factory)
    monkeypatch.setattr(stageb_fitting, "_fit_candidate_root_multi", _fake_fit_candidate_root_multi)
    monkeypatch.setattr(stageb_atom_mapping, "build_atom_to_leaf_map", _fake_build_atom_to_leaf_map)

    ds0 = _TinyDataset()
    ds1 = _TinyDataset()
    dl0 = torch.utils.data.DataLoader(ds0, batch_size=2, shuffle=False, drop_last=True)
    dl1 = torch.utils.data.DataLoader(ds1, batch_size=2, shuffle=False, drop_last=True)

    res = SimpleNamespace(
        residual_asts=[_dummy_residual(tag="c0", init=1.0), _dummy_residual(tag="c0", init=2.0)],
        dataset_ids=["d0", "d1"],
        term_asts=[None],
        coeffs=torch.tensor([[1.0], [2.0]], dtype=torch.float64),
    )

    _state, meta = _run_stageb_residual_refine_multi(
        res=res,
        surrogates=["s0", "s1"],
        ds_tr_list=[ds0, ds1],
        ds_va_list=[ds0, ds1],
        dl_tr_list=[dl0, dl1],
        dl_va_list=[dl0, dl1],
        lm_hp=SimpleNamespace(),
        cfg=DESearchConfig(),
        device=torch.device("cpu"),
        dtype=torch.float64,
        epochs_stageB=11,
    )

    assert captured["atom_factory"] == ["factory:s0", "factory:s1"]
    for yv in captured["val_targets"]:
        assert torch.allclose(yv, torch.zeros_like(yv))
    assert float(meta["val_mse"]) == 1.0
    assert [float(v) for v in meta["val_mse_per_dataset"]] == [1.0, 4.0]
    assert [float(v) for v in meta["val_rms_per_dataset"]] == [1.0, 2.0]
    assert int(meta["epochs"]) == 11
    assert bool(meta["coefficients_updated"]) is True
    assert [[float(v) for v in row] for row in res.coeffs.tolist()] == [[1.25], [2.5]]
