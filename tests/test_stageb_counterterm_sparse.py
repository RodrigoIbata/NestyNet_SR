# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

import torch

from nestynet_sr.sr_core.bridges import (
    AtomNode,
    ast_to_human_readable,
    collect_nn_atoms,
)
from nestynet_sr.sr_search.stageB import splits
from nestynet_sr.sr_search.stageB.splits import (
    _build_counterterm_mul_split_candidate,
    _fit_counterterm_polys_two_sided_for_mul_split,
)


def _counterterm_fixture():
    z = torch.linspace(1.0, 3.0, 30, dtype=torch.float64)
    t = torch.linspace(0.5, 2.5, 31, dtype=torch.float64)
    Z, T = torch.meshgrid(z, t, indexing="ij")
    zz = Z.reshape(-1)
    tt = T.reshape(-1)
    X = torch.stack([zz, tt], dim=1)

    # u(z,t) = -1/z + z^2*(t+2).  Subtracting the sparse reciprocal
    # counterterm leaves an exactly multiplicatively separable residual.
    u = -1.0 / zz + zz**2 * (tt + 2.0)
    du = torch.stack(
        [
            1.0 / zz**2 + 2.0 * zz * (tt + 2.0),
            zz**2,
        ],
        dim=1,
    )
    H = torch.zeros((X.shape[0], 2, 2), dtype=torch.float64)
    H[:, 0, 0] = -2.0 / zz**3 + 2.0 * (tt + 2.0)
    H[:, 0, 1] = 2.0 * zz
    H[:, 1, 0] = 2.0 * zz
    return X, u, du, H


def test_counterterm_sparse_reciprocal_power_fit():
    X, u, du, H = _counterterm_fixture()

    res = _fit_counterterm_polys_two_sided_for_mul_split(
        X=X,
        u=u,
        du=du,
        H=H,
        A=[0],
        B=[1],
        degree_A=1,
        degree_B=2,
        n_alt=3,
        variant="A_only",
        basis_A="power",
        power_A=-1,
    )

    assert res is not None
    assert res["rel_err"] < 1.0e-10
    assert res["basis_A"] == "power"
    assert res["power_A"] == -1
    assert torch.allclose(res["coeffs_A"], torch.tensor([-1.0], dtype=torch.float64))


def test_counterterm_sparse_candidate_keeps_nn_coordinate_uninverted(monkeypatch):
    X, u, du, H = _counterterm_fixture()
    target = AtomNode(kind="nn", var_idxs=(0, 1), kwargs={}, tag="leaf0")

    monkeypatch.setattr(
        splits,
        "_gather_nn_atom_value_grad_hess",
        lambda **_kwargs: (X, X, u, du, H),
    )

    cand_root, _init_fn, meta = _build_counterterm_mul_split_candidate(
        root=target,
        target=target,
        model=torch.nn.Identity(),
        reuse={},
        train_loader=[],
        device=torch.device("cpu"),
        dtype=torch.float64,
        degrees_A=(2,),
        degrees_B=(2,),
        rel_err_tol=2.0e-2,
    )

    assert cand_root is not None
    assert meta["counterterm_basis_A"] == "power"
    assert meta["counterterm_power_A"] == -1

    rendered = ast_to_human_readable(cand_root)
    assert "(x0)**-1" in rendered
    atoms = collect_nn_atoms(cand_root)
    rendered_atoms = [ast_to_human_readable(a) for a in atoms]
    assert "NN[x0]" in rendered_atoms
    assert "NN[x1]" in rendered_atoms
    assert all("x0)**-1" not in s for s in rendered_atoms)


def test_counterterm_sparse_init_handles_scalar_scale_parameter(monkeypatch):
    X, u, du, H = _counterterm_fixture()
    target = AtomNode(kind="nn", var_idxs=(0, 1), kwargs={}, tag="leaf0")

    monkeypatch.setattr(
        splits,
        "_gather_nn_atom_value_grad_hess",
        lambda **_kwargs: (X, X, u, du, H),
    )

    cand_root, init_fn, meta = _build_counterterm_mul_split_candidate(
        root=target,
        target=target,
        model=torch.nn.Identity(),
        reuse={},
        train_loader=[],
        device=torch.device("cpu"),
        dtype=torch.float64,
        degrees_A=(2,),
        degrees_B=(2,),
        rel_err_tol=2.0e-2,
    )

    assert cand_root is not None
    assert init_fn is not None
    assert meta["counterterm_basis_A"] == "power"
    assert meta["counterterm_power_A"] == -1

    class ScalarLeaf(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.value = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float64))

    scale_leaves = {}

    def fake_atom_map(root_new, _model_new):
        out = {}
        for atom in splits._collect_all_atoms(root_new):
            if isinstance(atom, AtomNode) and str(atom.kind).lower() == "scale":
                leaf = ScalarLeaf()
                scale_leaves[atom.tag] = leaf
                out[id(atom)] = leaf
        return out

    monkeypatch.setattr(splits, "build_atom_to_leaf_map", fake_atom_map)

    init_fn(cand_root, torch.nn.Module())

    assert scale_leaves
    assert any(
        torch.allclose(leaf.value.detach(), torch.tensor(-1.0, dtype=torch.float64))
        for leaf in scale_leaves.values()
    )
