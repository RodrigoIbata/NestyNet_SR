from types import SimpleNamespace

import numpy as np
import torch

from nestynet_sr.sr_core.bridges import (
    AtomNode,
    Var,
    eval_input_expr,
)
from nestynet_sr.sr_core.units import UnitSystem, UnitsSpec
from nestynet_sr.sr_gs import GeneralizedSymmetryConfig
from nestynet_sr.sr_search import search as _search_facade  # noqa: F401
from nestynet_sr.sr_search import _search_compounds as compounds
from nestynet_sr.sr_search import _search_runtime as runtime


class _AIF012Leaf(torch.nn.Module):
    def forward(self, x):
        radial = torch.sum(x[:, 1:4] ** 2, dim=1, keepdim=True)
        return 0.5 * x[:, 0:1] * radial

    def grad(self, cache):
        x = cache["x"]
        radial = torch.sum(x[:, 1:4] ** 2, dim=1, keepdim=True)
        return torch.cat(
            (
                0.5 * radial,
                x[:, 0:1] * x[:, 1:2],
                x[:, 0:1] * x[:, 2:3],
                x[:, 0:1] * x[:, 3:4],
            ),
            dim=1,
        ).unsqueeze(1)


def _cfg():
    return GeneralizedSymmetryConfig(
        enabled=True,
        mode="propose",
        policy="augment",
        general_affine=False,
        pairwise_composition=True,
        recursive_composition=True,
        recursive_composition_max_depth=3,
        recursive_composition_beam_width=2,
        max_stagea_proposals=12,
        stagea_proposal_budget=6,
    )


def _aif012_units():
    system = UnitSystem(("L", "T", "M", "I", "Theta"))
    return UnitsSpec(
        unit_system=system,
        x_dims=(
            system.dim([0, 0, 1, 0, 0]),
            system.dim([1, -1, 0, 0, 0]),
            system.dim([1, -1, 0, 0, 0]),
            system.dim([1, -1, 0, 0, 0]),
        ),
        y_dim=system.dim([2, -2, 1, 0, 0]),
    )


def _preflight_kwargs(*, atom, leaf, x, search_hp, units_spec):
    y = leaf(x)
    return {
        "model": object(),
        "current_ast": atom,
        "atom": atom,
        "tag_to_leaf": {atom.tag: leaf},
        "datagen_train_noshuffle": [(x, y)],
        "datagen_val_noshuffle": [(x, y)],
        "device": torch.device("cpu"),
        "dtype": torch.float64,
        "leaf_builder": None,
        "dual_layer_used": False,
        "search_hp": search_hp,
        "lm_hp": SimpleNamespace(),
        "loss_target_eff": 1.0e-8,
        "accept_threshold_eff_cand": 1.0e-6,
        "best_val_loss": 1.0e-7,
        "current_val_loss": 1.0e-7,
        "best_train_loss": 1.0e-7,
        "loss_scale": 1.0,
        "model_sep_output": None,
        "y_op": None,
        "y_op_inv": None,
        "Nxvars": 4,
        "x_transform_map": None,
        "trig_spec": None,
        "units_spec": units_spec,
        "enforce_units": True,
        "scaling_features": [],
    }


def test_pb012_root_gets_one_unit_certified_decisive_gs_preflight(monkeypatch):
    rng = np.random.default_rng(12)
    x = torch.as_tensor(rng.uniform(1.0, 5.0, size=(700, 4)), dtype=torch.float64)
    leaf = _AIF012Leaf()
    atom = AtomNode(
        kind="nn",
        var_idxs=(0, 1, 2, 3),
        kwargs={"num_segments": 8, "dual_layer": False},
        tag="pb012_root",
    )
    search_hp = SimpleNamespace(
        enable_compound_detection=True,
        compound_max_batches=1,
        gs_config=_cfg(),
        num_segments_map={False: 8, True: 8},
    )
    captured = []

    def _fake_try(**kwargs):
        captured.append(kwargs)
        return False, None, None, None, False, False

    # The facade refreshes late bindings while GS discovery runs, so patch both
    # the source module and its authoritative facade binding.
    monkeypatch.setattr(compounds, "_try_compound_candidates_for_atom", _fake_try)
    monkeypatch.setattr(_search_facade, "_try_compound_candidates_for_atom", _fake_try)
    kwargs = _preflight_kwargs(
        atom=atom,
        leaf=leaf,
        x=x,
        search_hp=search_hp,
        units_spec=_aif012_units(),
    )

    result = compounds._try_stageA_decisive_gs_preflight_for_atom(**kwargs)

    assert result[0] is False
    assert len(captured) == 1
    assert captured[0]["decisive_gs_only"] is True
    proposals = captured[0]["proposals"]
    assert len(proposals) == 1
    pattern, carrier_ast, confidence, extra, meta = proposals[0]
    assert tuple(pattern) == (1, 1, 1, 1)
    assert extra is None
    assert confidence >= 0.995
    assert meta["source"] == "generalized_symmetry"
    assert meta["carrier_certified"] is True
    assert meta["candidate_role"] == "inner_coordinate"
    assert meta["gs_carrier_depth"] == 2
    assert meta["gs_stagea_lane"] == "decisive"

    carrier = eval_input_expr(carrier_ast, x).detach().cpu().numpy().reshape(-1)
    target = leaf(x).detach().cpu().numpy().reshape(-1)
    assert abs(np.corrcoef(carrier, target)[0, 1]) > 1.0 - 1.0e-10

    # A failed decisive trial is remembered for this exact AST/carrier
    # transaction and is not trained again on a later Stage-A restart.
    result_retry = compounds._try_stageA_decisive_gs_preflight_for_atom(**kwargs)
    assert result_retry[0] is False
    assert len(captured) == 1


def test_preflight_respects_non_proposing_mode_and_zero_trial_budget(monkeypatch):
    rng = np.random.default_rng(120)
    x = torch.as_tensor(rng.uniform(1.0, 5.0, size=(300, 4)), dtype=torch.float64)
    leaf = _AIF012Leaf()
    atom = AtomNode(
        kind="nn",
        var_idxs=(0, 1, 2, 3),
        kwargs={"num_segments": 8, "dual_layer": False},
        tag="pb012_config_guard",
    )

    def _unexpected_trial(**_kwargs):
        raise AssertionError("disabled GS preflight entered the training lane")

    monkeypatch.setattr(
        compounds,
        "_try_compound_candidates_for_atom",
        _unexpected_trial,
    )
    monkeypatch.setattr(
        _search_facade,
        "_try_compound_candidates_for_atom",
        _unexpected_trial,
    )

    audit_cfg = _cfg()
    audit_cfg.mode = "audit"
    audit_hp = SimpleNamespace(
        enable_compound_detection=True,
        compound_max_batches=1,
        gs_config=audit_cfg,
        num_segments_map={False: 8, True: 8},
    )
    audit_result = compounds._try_stageA_decisive_gs_preflight_for_atom(
        **_preflight_kwargs(
            atom=atom,
            leaf=leaf,
            x=x,
            search_hp=audit_hp,
            units_spec=_aif012_units(),
        )
    )
    assert audit_result == (False, None, None, None, False, False)

    zero_budget_cfg = _cfg()
    zero_budget_cfg.decisive_stagea_max_trials = 0
    zero_budget_hp = SimpleNamespace(
        enable_compound_detection=True,
        compound_max_batches=1,
        gs_config=zero_budget_cfg,
        num_segments_map={False: 8, True: 8},
    )
    zero_budget_result = compounds._try_stageA_decisive_gs_preflight_for_atom(
        **_preflight_kwargs(
            atom=atom,
            leaf=leaf,
            x=x,
            search_hp=zero_budget_hp,
            units_spec=_aif012_units(),
        )
    )
    assert zero_budget_result == (False, None, None, None, False, False)

    limited_arity_hp = SimpleNamespace(
        enable_compound_detection=True,
        compound_max_batches=1,
        compound_max_vars=3,
        gs_config=_cfg(),
        num_segments_map={False: 8, True: 8},
    )
    limited_arity_result = compounds._try_stageA_decisive_gs_preflight_for_atom(
        **_preflight_kwargs(
            atom=atom,
            leaf=leaf,
            x=x,
            search_hp=limited_arity_hp,
            units_spec=_aif012_units(),
        )
    )
    assert limited_arity_result == (False, None, None, None, False, False)


def test_post_accept_bookkeeping_faults_cannot_escape_or_reopen_fallback():
    calls = []
    parent_ast = Var(0)
    candidate_ast = Var(1)

    def _details(*_args):
        calls.append("details")
        raise RuntimeError("details fault")

    def _record(**kwargs):
        calls.append("record")
        assert kwargs["details"] == {"full_compound": True}
        raise RuntimeError("record fault")

    def _sync(*_args, **_kwargs):
        calls.append("sync")
        raise RuntimeError("sync fault")

    runtime._stageA_record_decisive_gs_preflight_best_effort(
        candidate_model=object(),
        parent_ast=parent_ast,
        candidate_ast=candidate_ast,
        parent_loss=1.0,
        candidate_loss=0.1,
        full_compound=True,
        search_hp=SimpleNamespace(),
        move_details=_details,
        record_move=_record,
        sync_shadow=_sync,
    )

    assert calls == ["details", "record", "sync"]


def test_decisive_only_trial_does_not_inject_or_try_ordinary_proposals(monkeypatch):
    def _unexpected(*_args, **_kwargs):
        raise AssertionError("ordinary proposal injector entered decisive-only preflight")

    monkeypatch.setattr(
        compounds,
        "_stageA_append_compound_replay_proposals",
        _unexpected,
    )
    monkeypatch.setattr(
        compounds,
        "_stageA_append_visible_buckingham_1d_prefactor_proposals",
        _unexpected,
    )
    monkeypatch.setattr(
        compounds,
        "_stageA_append_noisy_soft_monomial_compound_proposals",
        _unexpected,
    )

    atom = AtomNode(
        kind="nn",
        var_idxs=(0, 1),
        kwargs={"num_segments": 8, "dual_layer": False},
        tag="low_confidence_gs",
    )
    proposal = (
        (1, 1),
        Var(0),
        0.5,
        None,
        {
            "source": "generalized_symmetry",
            "carrier_certified": True,
            "candidate_role": "inner_coordinate",
            "gs_carrier_fingerprint": "low-confidence",
        },
    )
    result = compounds._try_compound_candidates_for_atom(
        proposals=[proposal],
        model=object(),
        current_ast=atom,
        atom=atom,
        tag_to_leaf={atom.tag: object()},
        datagen_train_noshuffle=[],
        datagen_val_noshuffle=[],
        device=torch.device("cpu"),
        dtype=torch.float64,
        leaf_builder=None,
        dual_layer_used=False,
        search_hp=SimpleNamespace(
            gs_config=_cfg(),
            compound_max_proposals_to_try=6,
            num_segments_map={False: 8, True: 8},
        ),
        lm_hp=SimpleNamespace(),
        loss_target_eff=1.0e-8,
        accept_threshold_eff_cand=1.0e-6,
        best_val_loss=1.0e-7,
        current_val_loss=1.0e-7,
        best_train_loss=1.0e-7,
        loss_scale=1.0,
        model_sep_output=None,
        y_op=None,
        y_op_inv=None,
        Nxvars=2,
        x_transform_map=None,
        decisive_gs_only=True,
    )

    assert result == (False, None, None, None, False, False)


def test_later_compound_lane_does_not_retry_failed_preflight_gs_copy(monkeypatch):
    atom = AtomNode(
        kind="nn",
        var_idxs=(0, 1),
        kwargs={"num_segments": 8, "dual_layer": False},
        tag="attempted_gs",
    )
    proposal = (
        (1, 1),
        Var(0),
        1.0,
        None,
        {
            "source": "generalized_symmetry",
            "carrier_certified": True,
            "candidate_role": "inner_coordinate",
            "gs_carrier_fingerprint": "already-attempted",
        },
    )
    ordinary_copy = (
        (1, 1),
        proposal[1],
        0.9,
        None,
        {
            "source": "ordinary_detector",
            "kind": "monomial",
        },
    )
    search_hp = SimpleNamespace(
        gs_config=_cfg(),
        compound_max_proposals_to_try=6,
        num_segments_map={False: 8, True: 8},
    )
    search_hp._stageA_decisive_gs_preflight_attempted = {
        compounds._stageA_gs_preflight_attempt_key(atom, proposal)
    }

    monkeypatch.setattr(
        compounds,
        "_stageA_append_compound_replay_proposals",
        lambda proposals, **_kwargs: list(proposals),
    )
    monkeypatch.setattr(
        compounds,
        "_stageA_append_visible_buckingham_1d_prefactor_proposals",
        lambda proposals, **_kwargs: list(proposals),
    )
    monkeypatch.setattr(
        compounds,
        "_stageA_append_noisy_soft_monomial_compound_proposals",
        lambda proposals, **_kwargs: list(proposals),
    )
    scheduled = []

    def _capture_schedule(proposals, **_kwargs):
        scheduled.extend(proposals)
        return [], [], []

    monkeypatch.setattr(
        compounds,
        "_stageA_schedule_gs_compound_lanes",
        _capture_schedule,
    )

    result = compounds._try_compound_candidates_for_atom(
        proposals=[proposal, ordinary_copy],
        model=object(),
        current_ast=atom,
        atom=atom,
        tag_to_leaf={atom.tag: object()},
        datagen_train_noshuffle=[],
        datagen_val_noshuffle=[],
        device=torch.device("cpu"),
        dtype=torch.float64,
        leaf_builder=None,
        dual_layer_used=False,
        search_hp=search_hp,
        lm_hp=SimpleNamespace(),
        loss_target_eff=1.0e-8,
        accept_threshold_eff_cand=1.0e-6,
        best_val_loss=1.0e-7,
        current_val_loss=1.0e-7,
        best_train_loss=1.0e-7,
        loss_scale=1.0,
        model_sep_output=None,
        y_op=None,
        y_op_inv=None,
        Nxvars=2,
        x_transform_map=None,
    )

    assert result == (False, None, None, None, False, False)
    assert len(scheduled) == 1
    assert scheduled[0][4]["source"] == "ordinary_detector"
