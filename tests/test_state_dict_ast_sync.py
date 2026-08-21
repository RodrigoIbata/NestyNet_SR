# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

import nestynet
import torch

from nestynet_sr.sr_core.bridges import AtomNode, sync_ast_num_segments_from_state_dict
import pytest

from nestynet_sr.sr_search.stageB.engine import (
    _checkpoint_state_dict_cpu,
    _load_checkpoint_state_dict,
)


def test_sync_ast_nn_kwargs_from_single_layer_state_dict():
    ast = AtomNode("nn", (0,), kwargs={"num_segments": 16, "dual_layer": True})
    state_dict = {
        "leaf.0.model._ever_active": torch.ones(7),
        "leaf.0.base_model._ever_active": torch.ones(7),
    }

    out = sync_ast_num_segments_from_state_dict(ast, state_dict)

    assert out == {0: 7}
    assert ast.kwargs["num_segments"] == 7
    assert ast.kwargs["dual_layer"] is False


def test_sync_ast_nn_kwargs_from_dual_layer_state_dict():
    ast = AtomNode("nn", (0,), kwargs={"num_segments": 16, "dual_layer": False})
    state_dict = {
        "leaf.0._stage0.model._ever_active": torch.ones(5),
        "leaf.0._stage1.model._ever_active": torch.ones(5),
    }

    out = sync_ast_num_segments_from_state_dict(ast, state_dict)

    assert out == {0: 5}
    assert ast.kwargs["num_segments"] == 5
    assert ast.kwargs["dual_layer"] is True


def test_checkpoint_state_dict_does_not_refresh_or_mutate_live_model():
    class _Base(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.a_fit = torch.nn.Parameter(torch.ones(1))
            self.c_fit = None
            self.refreshed = False

        def _refresh_parameters_from_fixed(self):
            self.c_fit = torch.nn.Parameter(torch.full((1,), 2.0))
            self.refreshed = True

    class _Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.leaf = _Base()

    model = _Model()

    assert "leaf.c_fit" not in model.state_dict()
    state_dict = _checkpoint_state_dict_cpu(model)

    assert model.leaf.refreshed is False
    assert "leaf.c_fit" not in state_dict
    torch.testing.assert_close(state_dict["leaf.a_fit"], torch.ones(1))
    with torch.no_grad():
        model.leaf.a_fit.fill_(9.0)
    torch.testing.assert_close(state_dict["leaf.a_fit"], torch.ones(1))


def test_checkpoint_round_trip_preserves_effective_segmented_parameters():
    torch.manual_seed(1234)
    kwargs = dict(
        model_base_name="G_Model",
        Nout_size=1,
        Nx_size=1,
        num_segments=3,
        model_scale=0.1,
        dtype=torch.float64,
        device=torch.device("cpu"),
        seg_width=2,
    )
    model = nestynet.nets.NestyNet_Model(**kwargs)
    base = model.base_model

    # Block fitting can leave only a subset registered in each live fit
    # Parameter.  First preserve the complete initial state in fixed storage,
    # then make the selected fit coordinates deliberately authoritative.
    base.commit_fit_parameters_to_fixed_()
    indices = {}
    for group in base._parameter_group_names:
        pieces = getattr(base, f"{group}_pieces_fixed")
        count = sum(piece.numel() for piece in pieces)
        indices[group] = torch.arange(0, count, 2, dtype=torch.long)
    base.set_fitting_parameters(
        indices["a"], indices["b"], indices["c"], indices["K"]
    )
    with torch.no_grad():
        for offset, group in enumerate(base._parameter_group_names, start=1):
            fit = getattr(base, f"{group}_fit")
            if fit is not None and fit.numel():
                fit.add_(10.0 * offset)

    x = torch.linspace(-1.0, 1.0, 11, dtype=torch.float64).unsqueeze(-1)
    prediction_before = model(x).detach().clone()
    fit_before = {
        group: getattr(base, f"{group}_fit").detach().clone()
        for group in base._parameter_group_names
    }
    fixed_before = {
        group: [piece.detach().clone() for piece in getattr(base, f"{group}_pieces_fixed")]
        for group in base._parameter_group_names
    }
    effective_before = {
        group: torch.cat([piece.reshape(-1) for piece in pieces]).detach().clone()
        for group, pieces in zip(base._parameter_group_names, base.get_parameters())
    }

    state_dict = _checkpoint_state_dict_cpu(model)

    torch.testing.assert_close(model(x), prediction_before)
    for group in base._parameter_group_names:
        torch.testing.assert_close(getattr(base, f"{group}_fit"), fit_before[group])
        for actual, expected in zip(
            getattr(base, f"{group}_pieces_fixed"), fixed_before[group]
        ):
            torch.testing.assert_close(actual, expected)
        for prefix in ("base_model", "G_Model"):
            torch.testing.assert_close(
                state_dict[f"{prefix}.{group}_fit"], effective_before[group]
            )

    restored = nestynet.nets.NestyNet_Model(**kwargs)
    _load_checkpoint_state_dict(restored, state_dict)

    torch.testing.assert_close(restored(x), prediction_before)
    for group, pieces in zip(
        restored.base_model._parameter_group_names,
        restored.base_model.get_parameters(),
    ):
        restored_effective = torch.cat([piece.reshape(-1) for piece in pieces])
        torch.testing.assert_close(restored_effective, effective_before[group])


def test_checkpoint_load_repairs_missing_transient_fit_parameter():
    class _Leaf(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.a_fit = torch.nn.Parameter(torch.ones(1))
            self.c_fit = torch.nn.Parameter(torch.full((1,), 7.0))

    class _Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.leaf = _Leaf()

    model = _Model()
    state_dict = {
        "leaf.a_fit": torch.full((1,), 3.0),
        # Simulate an older checkpoint that did not contain this lazy view.
        # The tolerant restore should fill it from the rebuilt model rather
        # than crashing the whole Stage-B run.
        # "leaf.c_fit": intentionally omitted
    }

    _load_checkpoint_state_dict(model, state_dict)

    assert torch.allclose(model.leaf.a_fit, torch.tensor([3.0]))
    assert torch.allclose(model.leaf.c_fit, torch.tensor([7.0]))


def test_checkpoint_load_keeps_non_transient_mismatches_strict():
    class _Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))

    model = _Model()

    with pytest.raises(RuntimeError):
        _load_checkpoint_state_dict(model, {})
