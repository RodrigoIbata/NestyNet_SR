# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata

import torch

from nestynet_sr.sr_core.bridges import AtomNode
from nestynet_sr.sr_core.bridges import MulNode
from nestynet_sr.sr_search import candidate_builders as cb


def test_nls_candidate_refit_failure_never_emits_implicit_rational_support(monkeypatch):
    target = AtomNode(kind="nn", var_idxs=(0, 1), kwargs={}, tag="nls_refit_fail")

    def _fake_gather_atom_teacher_data(*args, **kwargs):
        x = torch.rand(
            600,
            2,
            generator=torch.Generator().manual_seed(1206),
            dtype=torch.float64,
        )
        return x, 1.0 + x[:, 0]

    monkeypatch.setattr(cb, "_gather_atom_teacher_data", _fake_gather_atom_teacher_data)
    monkeypatch.setattr(cb, "_fit_rational_coeffs_nd", lambda *args, **kwargs: None)

    result = cb._build_nonlinear_sub_candidate(
        root=target,
        target=target,
        reuse={"nls_refit_fail": torch.nn.Identity()},
        train_loader=[],
        device=torch.device("cpu"),
        dtype=torch.float64,
        hit={
            "col_idx": 0,
            "transform": "cos",
            "deg_num": 2,
            "deg_den": 2,
            "error": 1e-6,
        },
    )

    assert result is None


def test_nls_candidate_emits_sparse_ratpoly_overrides(monkeypatch):
    """NLS candidate should fall back to sparse plain ratpoly when pivot is ambiguous."""
    target = AtomNode(kind="nn", var_idxs=(0, 1), kwargs={}, tag="nls_test")

    def _fake_gather_atom_teacher_data(*args, **kwargs):
        gen = torch.Generator().manual_seed(123)
        x = torch.rand(600, 2, generator=gen, dtype=torch.float64)
        f = 1.0 + 0.2 * x[:, 0] - 0.1 * x[:, 1]
        return x, f

    def _fake_fit_rational_coeffs_nd(*args, **kwargs):
        a = torch.tensor([1.0, 0.8, -0.7], dtype=torch.float64)
        b = torch.tensor([1.0, 2.0, 0.25], dtype=torch.float64)
        support_num = torch.tensor([0, 3, 5], dtype=torch.int64)
        support_den = torch.tensor([0, 4, 8], dtype=torch.int64)
        return a, b, support_num, support_den

    monkeypatch.setattr(cb, "_gather_atom_teacher_data", _fake_gather_atom_teacher_data)
    monkeypatch.setattr(cb, "_fit_rational_coeffs_nd", _fake_fit_rational_coeffs_nd)

    hit = {
        "col_idx": 1,
        "transform": "cos",
        "deg_num": 3,
        "deg_den": 3,
        "error": 1e-6,
        "outer_transform": "identity",
    }

    result = cb._build_nonlinear_sub_candidate(
        root=target,
        target=target,
        reuse={"nls_test": torch.nn.Identity()},
        train_loader=[],
        device=torch.device("cpu"),
        dtype=torch.float64,
        hit=hit,
    )
    assert result is not None

    cand_root, _custom_init, meta = result
    assert isinstance(cand_root, AtomNode)
    assert cand_root.kind.lower() == "ratpoly"
    # Numerator support max-degree is 2, so candidate compresses 3 -> 2.
    assert cand_root.kwargs["deg_num"] == 2
    # Denominator support contains degree-3 term (idx=8), so stays at 3.
    assert cand_root.kwargs["deg_den"] == 3
    assert cand_root.kwargs["exps_num_override"] == [[0, 0], [0, 2], [2, 0]]
    assert cand_root.kwargs["exps_den_override"] == [[0, 0], [1, 1], [2, 1]]
    assert int(meta["n_terms_num"]) == 3
    assert int(meta["n_terms_den"]) == 3
    assert int(meta["deg_num_screen"]) == 3


def test_nls_candidate_emits_scale_times_rratpoly_when_clear_nd_pivot(monkeypatch):
    target = AtomNode(kind="nn", var_idxs=(0, 1), kwargs={}, tag="nls_test")

    def _fake_gather_atom_teacher_data(*args, **kwargs):
        gen = torch.Generator().manual_seed(123)
        x = torch.rand(600, 2, generator=gen, dtype=torch.float64)
        f = 1.0 + 0.2 * x[:, 0] - 0.1 * x[:, 1]
        return x, f

    def _fake_fit_rational_coeffs_nd(*args, **kwargs):
        a = torch.tensor([1.5, -0.25], dtype=torch.float64)
        b = torch.tensor([1.0, 2.0, 0.25], dtype=torch.float64)
        support_num = torch.tensor([0, 5], dtype=torch.int64)
        support_den = torch.tensor([0, 4, 8], dtype=torch.int64)
        return a, b, support_num, support_den

    monkeypatch.setattr(cb, "_gather_atom_teacher_data", _fake_gather_atom_teacher_data)
    monkeypatch.setattr(cb, "_fit_rational_coeffs_nd", _fake_fit_rational_coeffs_nd)

    hit = {
        "col_idx": 1,
        "transform": "cos",
        "deg_num": 3,
        "deg_den": 3,
        "error": 1e-6,
        "outer_transform": "identity",
    }

    result = cb._build_nonlinear_sub_candidate(
        root=target,
        target=target,
        reuse={"nls_test": torch.nn.Identity()},
        train_loader=[],
        device=torch.device("cpu"),
        dtype=torch.float64,
        hit=hit,
    )
    assert result is not None

    cand_root, _custom_init, meta = result
    assert isinstance(cand_root, MulNode)
    assert cand_root.left.kind.lower() == "scale"
    assert cand_root.right.kind.lower() == "rratpoly"
    assert cand_root.right.kwargs["deg_num"] == 2
    assert cand_root.right.kwargs["deg_den"] == 3
    assert cand_root.right.kwargs["exps_num_override"] == [[0, 0], [2, 0]]
    assert cand_root.right.kwargs["exps_den_override"] == [[0, 0], [1, 1], [2, 1]]
    assert meta["leaf_kind"] == "rratpoly"
    assert meta["reduced"] is True
