# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

import math
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from nestynet_sr.sr_core import collect_nn_atoms
from nestynet_sr.sr_core.bridges import AcosNode, AddNode, AsinNode, AtomNode, ConstNode, MulNode, PowNode, Var
from nestynet_sr.sr_core.units import UnitSystem, UnitsSpec
from nestynet_sr.sr_search.representation import pretty_print_state
from nestynet_sr.sr_search.phase_scan import (
    OmegaTerm,
    OuterLinkHint,
    PhaseContextHint,
    PhaseHint,
    PhaseScanHyperparams,
    stable_int_hash,
    phase_hint_omega_candidates,
    run_outer_inverse_trig_prescan,
    run_phase_context_scan,
    run_phase_prescan,
)
from nestynet_sr.sr_search.stageB.engine import StageBContext, StageBState
from nestynet_sr.sr_search.stageB.rules import (
    RuleLastHardTrigSquare1D,
    RuleInverseTrigOuterClosure,
    RuleInverseTrigRationalOuterClosure,
    RulePhaseContextTrigClosure,
    RulePhaseHintTrigClosure,
    RulePhaseHintReciprocalTrigPower,
    RuleOverlapPrefactorPeelNN,
    _stageB_phase_hints_for_atom,
)


class _SinSquareTeacher(torch.nn.Module):
    def __init__(self, omega: float):
        super().__init__()
        self.omega = float(omega)

    def forward(self, x):
        z = x.reshape(x.shape[0], -1)[:, :1]
        return torch.sin(self.omega * z) ** 2


def test_phase_prescan_finds_dimensionless_product_ratio():
    rng = np.random.default_rng(123)
    x0 = rng.uniform(1.0, 2.0, size=2400)
    x1 = rng.uniform(1.0, 2.0, size=2400)
    x2 = rng.uniform(1.0, 4.0, size=2400)
    X = np.stack([x0, x1, x2], axis=1)
    y = np.sin(2.0 * math.pi * x0 * x1 / x2) ** 2

    us = UnitSystem(base=("M", "L", "T", "Q", "K"))
    units_payload = {
        "x_dims": tuple(
            us.dim(u)
            for u in (
                [2, -2, 1, 0, 0],
                [0, 1, 0, 0, 0],
                [2, -1, 1, 0, 0],
            )
        )
    }

    hints = run_phase_prescan(
        X,
        y,
        Nxvars=3,
        units_payload=units_payload,
        ignore_units=False,
        hp=PhaseScanHyperparams(sample_size=2000, max_candidates=96),
    )

    assert hints
    assert hints[0].carrier_label == "((x0 * x1) * (x2)**-1)"
    assert hints[0].unit_status == "dimensionless"
    assert hints[0].r2_phase > 0.99


def test_phase_hint_square_grid_keeps_sin_square_ambiguity():
    hint = PhaseHint(
        carrier_ast=None,
        carrier_label="z",
        phase_family="linear",
        observed_omega=2.0 * math.pi,
        carrier_omega_candidates=(2.0 * math.pi, math.pi),
        waveform_family="fourier",
        envelope_family="none",
        score=1.0,
        confidence=1.0,
        r2_phase=1.0,
        r2_trend=0.0,
        support_fraction=1.0,
        n_cycles=3.0,
        unit_status="dimensionless",
    )

    seeds = phase_hint_omega_candidates([hint], for_square=True)

    assert any(abs(w - 2.0 * math.pi) < 1.0e-12 for w in seeds)
    assert any(abs(w - 4.0 * math.pi) < 1.0e-12 for w in seeds)


def test_phase_prescan_records_explicit_omega_terms():
    rng = np.random.default_rng(321)
    x0 = rng.uniform(1.0, 2.0, size=2200)
    x1 = rng.uniform(1.0, 2.0, size=2200)
    x2 = rng.uniform(1.0, 4.0, size=2200)
    X = np.stack([x0, x1, x2], axis=1)
    y = np.sin(2.0 * math.pi * x0 * x1 / x2) ** 2

    hints = run_phase_prescan(
        X,
        y,
        Nxvars=3,
        units_payload=None,
        ignore_units=True,
        hp=PhaseScanHyperparams(sample_size=1800, max_candidates=96),
    )

    assert hints
    terms = tuple(getattr(hints[0], "omega_terms", ()) or ())
    assert terms
    assert any(t.family == "fourier" and t.harmonic == 1 for t in terms)
    assert any(t.family == "sin_square" and t.harmonic == 2 for t in terms)


def test_phase_stable_int_hash_is_process_stable():
    local = stable_int_hash("phase_hint_trig_closure", "z", "sin")
    code = (
        "from nestynet_sr.sr_search.phase_scan import stable_int_hash; "
        "print(stable_int_hash('phase_hint_trig_closure', 'z', 'sin'))"
    )
    other = int(subprocess.check_output([sys.executable, "-c", code], text=True).strip())

    assert local == other


def test_phase_context_scan_finds_pb094_prefactor_context():
    rng = np.random.default_rng(789)
    x0 = rng.uniform(1.0, 3.0, size=2600)
    x1 = rng.uniform(0.4, 3.0, size=2600)
    x2 = rng.uniform(0.5, 3.2, size=2600)
    X = np.stack([x0, x1, x2], axis=1)
    y = 2.0 * x0 * (1.0 - np.cos(x1 * x2))

    hints = run_phase_context_scan(
        X,
        y,
        Nxvars=3,
        units_payload=None,
        ignore_units=True,
        hp=PhaseScanHyperparams(sample_size=2200, max_support=2, max_candidates=80),
    )

    match = [
        h
        for h in hints
        if h.carrier_label == "(x1 * x2)"
        and "x0" in tuple((h.details or {}).get("context_labels", ()))
    ]
    assert match
    assert match[0].delta_r2_phase > 0.8
    assert match[0].waveform_family in {"one_minus_cos", "contextual_fourier"}


def test_phase_context_scan_rejects_unitful_phase_carrier():
    rng = np.random.default_rng(790)
    x0 = rng.uniform(1.0, 3.0, size=2200)
    x1 = rng.uniform(0.4, 3.0, size=2200)
    x2 = rng.uniform(0.5, 3.2, size=2200)
    X = np.stack([x0, x1, x2], axis=1)
    y = 2.0 * x0 * (1.0 - np.cos(x1 * x2))

    us = UnitSystem(base=("M", "L", "T", "Q", "K"))
    units_payload = {
        "y_dim": us.dim([1, 0, 0, 0, 0]),
        "x_dims": (
            us.dim([1, 0, 0, 0, 0]),
            us.dim([0, 1, 0, 0, 0]),
            us.dim([0, 1, 0, 0, 0]),
        ),
    }

    hints = run_phase_context_scan(
        X,
        y,
        Nxvars=3,
        units_payload=units_payload,
        ignore_units=False,
        hp=PhaseScanHyperparams(sample_size=1800, max_support=2, max_candidates=80),
    )

    assert all(h.carrier_label != "(x1 * x2)" for h in hints)


def test_outer_inverse_trig_prescan_finds_arcsin_product_ratio():
    rng = np.random.default_rng(792)
    x0 = rng.uniform(0.2, 0.9, size=2200)
    x1 = rng.uniform(0.3, 1.1, size=2200)
    x2 = rng.uniform(1.2, 2.4, size=2200)
    X = np.stack([x0, x1, x2], axis=1)
    y = np.arcsin(0.7 * x0 * x1 / x2)

    us = UnitSystem(base=("M", "L", "T", "Q", "K"))
    units_payload = {
        "y_dim": us.dimless(),
        "x_dims": tuple(
            us.dim(u)
            for u in (
                [2, -2, 1, 0, 0],
                [0, 1, 0, 0, 0],
                [2, -1, 1, 0, 0],
            )
        ),
    }

    hints = run_outer_inverse_trig_prescan(
        X,
        y,
        Nxvars=3,
        units_payload=units_payload,
        ignore_units=False,
        hp=PhaseScanHyperparams(sample_size=1800, max_candidates=96),
    )

    match = [
        h
        for h in hints
        if h.link_name == "arcsin"
        and h.transform_name == "sin"
        and h.carrier_label == "((x0 * x1) * (x2)**-1)"
    ]
    assert match
    assert match[0].r2 > 0.999
    assert match[0].rms_rel < 1.0e-6
    assert match[0].domain_ok_frac > 0.99
    assert match[0].branch_ok_frac > 0.99


def test_outer_inverse_trig_prescan_finds_arcsin_mixed_trig_carrier():
    rng = np.random.default_rng(795)
    x0 = rng.uniform(0.2, 0.9, size=2200)
    x1 = rng.uniform(0.2, 1.2, size=2200)
    X = np.stack([x0, x1], axis=1)
    y = np.arcsin(x0 * np.sin(x1))

    us = UnitSystem(base=("M", "L", "T", "Q", "K"))
    hints = run_outer_inverse_trig_prescan(
        X,
        y,
        Nxvars=2,
        units_payload={"y_dim": us.dimless(), "x_dims": (us.dimless(), us.dimless())},
        ignore_units=False,
        hp=PhaseScanHyperparams(sample_size=1800, max_support=2, max_candidates=48),
    )

    match = [
        h
        for h in hints
        if h.link_name == "arcsin"
        and h.transform_name == "sin"
        and h.carrier_label == "(x0 * sin(x1))"
    ]
    assert match
    assert match[0].r2 > 0.999
    assert match[0].rms_rel < 1.0e-6
    assert abs(match[0].affine_a - 1.0) < 1.0e-8
    assert abs(match[0].affine_b) < 1.0e-8
    assert match[0].domain_ok_frac > 0.99
    assert match[0].branch_ok_frac > 0.99


def test_outer_inverse_trig_prescan_finds_arcsin_affine_sin_carrier():
    rng = np.random.default_rng(798)
    x0 = rng.uniform(0.2, 0.9, size=2600)
    x1 = rng.uniform(0.1, 1.3, size=2600)
    X = np.stack([x0, x1], axis=1)
    y = np.arcsin(x0 * np.sin(2.0 * x1 + 0.3))

    us = UnitSystem(base=("M", "L", "T", "Q", "K"))
    hints = run_outer_inverse_trig_prescan(
        X,
        y,
        Nxvars=2,
        units_payload={"y_dim": us.dimless(), "x_dims": (us.dimless(), us.dimless())},
        ignore_units=False,
        hp=PhaseScanHyperparams(sample_size=2200, max_support=2, max_candidates=48),
    )

    match = [
        h
        for h in hints
        if h.link_name == "arcsin"
        and h.transform_name == "sin"
        and bool((h.details or {}).get("affine_trig", False))
        and (h.details or {}).get("trig_kind") == "sin"
        and int((h.details or {}).get("axis", -1)) == 1
        and abs(float((h.details or {}).get("omega", 0.0)) - 2.0) < 1.0e-12
    ]
    assert match
    assert match[0].r2 > 0.999
    assert match[0].rms_rel < 1.0e-6
    assert abs(float((match[0].details or {}).get("phase", 0.0)) - 0.3) < 1.0e-8
    assert abs(match[0].affine_a - 1.0) < 1.0e-8
    assert abs(match[0].affine_b) < 1.0e-8


def test_outer_inverse_trig_prescan_finds_arcsin_affine_cos_carrier():
    rng = np.random.default_rng(799)
    x0 = rng.uniform(0.2, 0.9, size=2600)
    x1 = rng.uniform(0.1, 1.3, size=2600)
    X = np.stack([x0, x1], axis=1)
    y = np.arcsin(x0 * np.cos(1.5 * x1 - 0.4))

    us = UnitSystem(base=("M", "L", "T", "Q", "K"))
    hints = run_outer_inverse_trig_prescan(
        X,
        y,
        Nxvars=2,
        units_payload={"y_dim": us.dimless(), "x_dims": (us.dimless(), us.dimless())},
        ignore_units=False,
        hp=PhaseScanHyperparams(sample_size=2200, max_support=2, max_candidates=48),
    )

    match = [
        h
        for h in hints
        if h.link_name == "arcsin"
        and h.transform_name == "sin"
        and bool((h.details or {}).get("affine_trig", False))
        and (h.details or {}).get("trig_kind") == "cos"
        and int((h.details or {}).get("axis", -1)) == 1
        and abs(float((h.details or {}).get("omega", 0.0)) - 1.5) < 1.0e-12
    ]
    assert match
    assert match[0].r2 > 0.999
    assert match[0].rms_rel < 1.0e-6
    assert abs(float((match[0].details or {}).get("phase", 0.0)) + 0.4) < 1.0e-8
    assert abs(match[0].affine_a - 1.0) < 1.0e-8
    assert abs(match[0].affine_b) < 1.0e-8


def test_outer_inverse_trig_prescan_rejects_unitful_target():
    rng = np.random.default_rng(793)
    x0 = rng.uniform(0.2, 0.9, size=1200)
    x1 = rng.uniform(0.3, 1.1, size=1200)
    x2 = rng.uniform(1.2, 2.4, size=1200)
    X = np.stack([x0, x1, x2], axis=1)
    y = np.arcsin(0.7 * x0 * x1 / x2)

    us = UnitSystem(base=("M", "L", "T", "Q", "K"))
    hints = run_outer_inverse_trig_prescan(
        X,
        y,
        Nxvars=3,
        units_payload={"y_dim": us.dim([1, 0, 0, 0, 0]), "x_dims": (us.dimless(),) * 3},
        ignore_units=False,
        hp=PhaseScanHyperparams(sample_size=1000, max_candidates=24),
    )

    assert hints == []


def test_outer_inverse_trig_prescan_rejects_unitful_mixed_trig_axis():
    rng = np.random.default_rng(797)
    x0 = rng.uniform(0.2, 0.9, size=1600)
    x1 = rng.uniform(0.2, 1.2, size=1600)
    X = np.stack([x0, x1], axis=1)
    y = np.arcsin(x0 * np.sin(x1))

    us = UnitSystem(base=("M", "L", "T", "Q", "K"))
    hints = run_outer_inverse_trig_prescan(
        X,
        y,
        Nxvars=2,
        units_payload={"y_dim": us.dimless(), "x_dims": (us.dimless(), us.dim([0, 1, 0, 0, 0]))},
        ignore_units=False,
        hp=PhaseScanHyperparams(sample_size=1400, max_support=2, max_candidates=48),
    )

    assert all("sin(x1)" not in h.carrier_label for h in hints)
    assert all("cos(x1)" not in h.carrier_label for h in hints)
    assert all(not (bool((h.details or {}).get("affine_trig", False)) and int((h.details or {}).get("axis", -1)) == 1) for h in hints)


def test_phase_hint_direct_trig_closure_proposes_visible_ast():
    rng = np.random.default_rng(456)
    x0 = rng.uniform(1.0, 2.0, size=1600)
    x1 = rng.uniform(1.0, 2.0, size=1600)
    x2 = rng.uniform(1.0, 4.0, size=1600)
    X = np.stack([x0, x1, x2], axis=1)
    y = np.sin(2.0 * math.pi * x0 * x1 / x2) ** 2

    carrier = MulNode(MulNode(Var(0), Var(1)), PowNode(Var(2), -1.0))
    hint = PhaseHint(
        carrier_ast=carrier,
        carrier_label="((x0 * x1) * (x2)**-1)",
        phase_family="linear",
        observed_omega=2.0 * math.pi,
        carrier_omega_candidates=(2.0 * math.pi, math.pi),
        waveform_family="fourier",
        envelope_family="none",
        score=1.0,
        confidence=1.0,
        r2_phase=1.0,
        r2_trend=0.0,
        support_fraction=1.0,
        n_cycles=3.0,
        unit_status="unchecked",
    )

    root = AtomNode(kind="nn", var_idxs=(0, 1, 2), tag="nn0")
    loader = DataLoader(
        TensorDataset(
            torch.as_tensor(X, dtype=torch.float64),
            torch.as_tensor(y.reshape(-1, 1), dtype=torch.float64),
        ),
        batch_size=512,
        shuffle=False,
    )
    state = StageBState(
        root=root,
        model=torch.nn.Identity(),
        reuse={},
        val_loss=1.0,
    )
    ctx = StageBContext(
        state=state,
        train_loader=loader,
        val_loader=loader,
        lm_hp=object(),
        device=torch.device("cpu"),
        dtype=torch.float64,
        epochs_stageB=1,
        loss_scale=1.0,
        loss_good_enough_raw=1.0e-9,
        score_tol=0.0,
        scale_specs=[],
        scaling_by_axis={},
        trig_by_axis={},
        phase_hints=[hint],
        verbose=False,
    )

    cands = RulePhaseHintTrigClosure().propose(ctx, root)

    assert cands
    assert any("square" in cand.label for cand in cands)
    assert all(not collect_nn_atoms(cand.root) for cand in cands if cand.root is not None)
    assert all(isinstance((cand.meta or {}).get("signature"), tuple) for cand in cands)


def test_outer_inverse_trig_direct_closure_proposes_visible_ast():
    rng = np.random.default_rng(794)
    x0 = rng.uniform(0.2, 0.9, size=1600)
    x1 = rng.uniform(0.3, 1.1, size=1600)
    x2 = rng.uniform(1.2, 2.4, size=1600)
    X = np.stack([x0, x1, x2], axis=1)
    y = np.arcsin(0.7 * x0 * x1 / x2)

    carrier = MulNode(MulNode(Var(0), Var(1)), PowNode(Var(2), -1.0))
    hint = OuterLinkHint(
        link_name="arcsin",
        transform_name="sin",
        carrier_ast=carrier,
        carrier_label="((x0 * x1) * (x2)**-1)",
        affine_a=0.7,
        affine_b=0.0,
        rms_rel=0.0,
        r2=1.0,
        domain_ok_frac=1.0,
        branch_ok_frac=1.0,
        confidence=1.0,
        unit_status="unchecked",
    )

    root = AtomNode(kind="nn", var_idxs=(0, 1, 2), tag="nn0")
    loader = DataLoader(
        TensorDataset(
            torch.as_tensor(X, dtype=torch.float64),
            torch.as_tensor(y.reshape(-1, 1), dtype=torch.float64),
        ),
        batch_size=512,
        shuffle=False,
    )
    state = StageBState(
        root=root,
        model=torch.nn.Identity(),
        reuse={},
        val_loss=1.0,
    )
    ctx = StageBContext(
        state=state,
        train_loader=loader,
        val_loader=loader,
        lm_hp=object(),
        device=torch.device("cpu"),
        dtype=torch.float64,
        epochs_stageB=1,
        loss_scale=1.0,
        loss_good_enough_raw=1.0e-9,
        score_tol=0.0,
        scale_specs=[],
        scaling_by_axis={},
        trig_by_axis={},
        outer_link_hints=[hint],
        verbose=False,
    )

    cands = RuleInverseTrigOuterClosure().propose(ctx, root)

    assert cands
    assert any(cand.label == "inverse_trig_outer_arcsin" for cand in cands)
    assert all(not collect_nn_atoms(cand.root) for cand in cands if cand.root is not None)
    assert all((cand.meta or {}).get("pattern") == "inverse_trig_outer_closure" for cand in cands)


def test_outer_inverse_trig_rational_closure_finds_pb109_shape_after_compound():
    rng = np.random.default_rng(109)
    x0 = rng.uniform(4.0, 6.0, size=2200)
    x1 = rng.uniform(1.0, 3.0, size=2200)
    x2 = rng.uniform(1.0, 3.0, size=2200)
    r = x1 / x0
    c = np.cos(x2)
    arg = (c - r) / (1.0 - r * c)
    y = np.arccos(np.clip(arg, -1.0, 1.0))
    X = np.stack([x0, x1, x2], axis=1)

    # This mirrors the useful Stage-A move in pb109: the current NN atom has
    # effective inputs z=x0/x1 and x2.  The rational outer-link rule must be
    # able to recover r=1/z and c=cos(x2) dynamically at Stage B time.
    z = MulNode(Var(0), PowNode(Var(1), -1.0))
    root = AtomNode(kind="nn", var_idxs=(0, 1, 2), tag="nn0", inputs=(z, Var(2)))
    loader = DataLoader(
        TensorDataset(
            torch.as_tensor(X, dtype=torch.float64),
            torch.as_tensor(y.reshape(-1, 1), dtype=torch.float64),
        ),
        batch_size=512,
        shuffle=False,
    )
    state = StageBState(root=root, model=torch.nn.Identity(), reuse={}, val_loss=1.0)
    ctx = StageBContext(
        state=state,
        train_loader=loader,
        val_loader=loader,
        lm_hp=SimpleNamespace(
            stageB_last_hard_trig_power_screen_rel_rms=2.0e-2,
            stageB_last_hard_trig_power_max_points=4096,
            macro_domain_ok_frac=0.98,
        ),
        device=torch.device("cpu"),
        dtype=torch.float64,
        epochs_stageB=1,
        loss_scale=1.0,
        loss_good_enough_raw=1.0e-9,
        score_tol=0.0,
        scale_specs=[],
        scaling_by_axis={},
        trig_by_axis={},
        verbose=False,
    )

    cands = RuleInverseTrigRationalOuterClosure().propose(ctx, root)

    assert cands
    best = cands[0]
    assert best.label == "inverse_trig_outer_rational_arccos_deg2"
    assert isinstance(best.root, AcosNode)
    assert all(not collect_nn_atoms(cand.root) for cand in cands if cand.root is not None)
    assert (best.meta or {}).get("pattern") == "inverse_trig_outer_rational_closure"
    assert (best.meta or {}).get("rational_degree") == 2
    assert any("cos(x2)" in str(v) for v in (best.meta or {}).get("feature_labels", ()))


def test_stageB_overlap_walkers_accept_inverse_trig_wrappers():
    X = np.ones((8, 3), dtype=np.float64)
    y = np.zeros((8, 1), dtype=np.float64)
    left = AtomNode(kind="nn", var_idxs=(0, 1), tag="nn0")
    right = AtomNode(kind="nn", var_idxs=(1, 2), tag="nn1")
    root = AsinNode(AddNode(left, right))
    loader = DataLoader(
        TensorDataset(
            torch.as_tensor(X, dtype=torch.float64),
            torch.as_tensor(y, dtype=torch.float64),
        ),
        batch_size=8,
        shuffle=False,
    )
    state = StageBState(root=root, model=torch.nn.Identity(), reuse={}, val_loss=1.0)
    ctx = StageBContext(
        state=state,
        train_loader=loader,
        val_loader=loader,
        lm_hp=object(),
        device=torch.device("cpu"),
        dtype=torch.float64,
        epochs_stageB=1,
        loss_scale=1.0,
        loss_good_enough_raw=1.0e-9,
        score_tol=0.0,
        scale_specs=[],
        scaling_by_axis={},
        trig_by_axis={},
        verbose=False,
    )

    targets = RuleOverlapPrefactorPeelNN().iter_targets(ctx)

    assert targets == [root.arg]


def test_pretty_print_state_accepts_inverse_trig_root():
    state = SimpleNamespace(
        root=AsinNode(ConstNode(0.5)),
        model=SimpleNamespace(leaf=[]),
    )

    assert pretty_print_state(state) == "asin(0.5)"


def test_phase_hint_reciprocal_trig_power_proposes_visible_ast():
    rng = np.random.default_rng(796)
    x0 = rng.uniform(0.2, 0.9, size=1800)
    x1 = rng.uniform(0.3, 1.1, size=1800)
    x2 = rng.uniform(1.4, 2.6, size=1800)
    X = np.stack([x0, x1, x2], axis=1)
    z = x0 * x1 / x2
    omega = 1.2
    y = 3.0 * np.sin(omega * z) ** -4

    carrier = MulNode(MulNode(Var(0), Var(1)), PowNode(Var(2), -1.0))
    hint = PhaseHint(
        carrier_ast=carrier,
        carrier_label="((x0 * x1) * (x2)**-1)",
        phase_family="linear",
        observed_omega=omega,
        carrier_omega_candidates=(omega,),
        waveform_family="fourier",
        envelope_family="none",
        score=1.0,
        confidence=1.0,
        r2_phase=1.0,
        r2_trend=0.0,
        support_fraction=1.0,
        n_cycles=1.0,
        unit_status="unchecked",
    )

    root = AtomNode(kind="nn", var_idxs=(0, 1, 2), tag="nn0")
    loader = DataLoader(
        TensorDataset(
            torch.as_tensor(X, dtype=torch.float64),
            torch.as_tensor(y.reshape(-1, 1), dtype=torch.float64),
        ),
        batch_size=512,
        shuffle=False,
    )
    state = StageBState(
        root=root,
        model=torch.nn.Identity(),
        reuse={},
        val_loss=1.0,
    )
    ctx = StageBContext(
        state=state,
        train_loader=loader,
        val_loader=loader,
        lm_hp=SimpleNamespace(
            stageB_last_hard_trig_power_screen_rel_rms=2.0e-2,
            stageB_last_hard_trig_power_max_offset_rel=0.15,
        ),
        device=torch.device("cpu"),
        dtype=torch.float64,
        epochs_stageB=1,
        loss_scale=1.0,
        loss_good_enough_raw=1.0e-9,
        score_tol=0.0,
        scale_specs=[],
        scaling_by_axis={},
        trig_by_axis={},
        phase_hints=[hint],
        verbose=False,
    )

    cands = RulePhaseHintReciprocalTrigPower().propose(ctx, root)

    assert cands
    assert any((cand.meta or {}).get("trig_power") == -4 for cand in cands)
    assert all(not collect_nn_atoms(cand.root) for cand in cands if cand.root is not None)
    assert all((cand.meta or {}).get("pattern") == "phase_hint_reciprocal_trig_power" for cand in cands)


def test_phase_context_direct_one_minus_cos_proposes_visible_ast():
    rng = np.random.default_rng(791)
    x0 = rng.uniform(1.0, 3.0, size=1800)
    x1 = rng.uniform(0.4, 3.0, size=1800)
    x2 = rng.uniform(0.5, 3.2, size=1800)
    X = np.stack([x0, x1, x2], axis=1)
    y = 2.0 * x0 * (1.0 - np.cos(x1 * x2))

    carrier = MulNode(Var(1), Var(2))
    hint = PhaseContextHint(
        carrier_ast=carrier,
        carrier_label="(x1 * x2)",
        phase_family="linear",
        omega_terms=(
            OmegaTerm(
                base_omega=1.0,
                harmonic=1,
                actual_omega=1.0,
                energy=1.0,
                family="fourier",
            ),
        ),
        context_asts=(Var(0),),
        coupling_mode="prefactor",
        waveform_family="one_minus_cos",
        r2_context_only=0.2,
        r2_context_phase=1.0,
        delta_r2_phase=0.8,
        confidence=1.0,
        unit_status="unchecked",
        details={"omega": 1.0, "support_fraction": 1.0, "context_labels": ("x0",)},
    )

    root = AtomNode(kind="nn", var_idxs=(0, 1, 2), tag="nn0")
    loader = DataLoader(
        TensorDataset(
            torch.as_tensor(X, dtype=torch.float64),
            torch.as_tensor(y.reshape(-1, 1), dtype=torch.float64),
        ),
        batch_size=512,
        shuffle=False,
    )
    state = StageBState(
        root=root,
        model=torch.nn.Identity(),
        reuse={},
        val_loss=1.0,
    )
    ctx = StageBContext(
        state=state,
        train_loader=loader,
        val_loader=loader,
        lm_hp=object(),
        device=torch.device("cpu"),
        dtype=torch.float64,
        epochs_stageB=1,
        loss_scale=1.0,
        loss_good_enough_raw=1.0e-9,
        score_tol=0.0,
        scale_specs=[],
        scaling_by_axis={},
        trig_by_axis={},
        phase_context_hints=[hint],
        verbose=False,
    )

    cands = RulePhaseContextTrigClosure().propose(ctx, root)

    assert cands
    assert any(cand.label == "phase_context_one_minus_cos" for cand in cands)
    assert all(not collect_nn_atoms(cand.root) for cand in cands if cand.root is not None)
    assert all((cand.meta or {}).get("pattern") == "phase_context_trig_closure" for cand in cands)


def test_stageB_local_phase_scan_seeds_last_hard_trig_square():
    n = 2200
    z = np.linspace(0.0, 1.0, n, dtype=np.float64)
    # Pick a non-default frequency aligned with the PhaseScan FFT grid, so the
    # local atom scan is what supplies the useful seed.
    harmonic_omega = 2.0 * math.pi * 3.0 * 383.0 / 384.0
    arg_omega = 0.5 * harmonic_omega
    X = z.reshape(-1, 1)
    y = np.sin(arg_omega * z) ** 2

    root = AtomNode(kind="nn", var_idxs=(0,), tag="nn0")
    loader = DataLoader(
        TensorDataset(
            torch.as_tensor(X, dtype=torch.float64),
            torch.as_tensor(y.reshape(-1, 1), dtype=torch.float64),
        ),
        batch_size=512,
        shuffle=False,
    )
    us = UnitSystem(base=("M", "L", "T", "Q", "K"))
    units_spec = UnitsSpec(
        unit_system=us,
        x_dims=(us.dimless(),),
        y_dim=us.dimless(),
        y_transform_name="identity",
        policy="free_const_only",
        nn_semantics="unknown",
    )
    state = StageBState(
        root=root,
        model=torch.nn.Identity(),
        reuse={"nn0": _SinSquareTeacher(arg_omega)},
        val_loss=1.0,
    )
    ctx = StageBContext(
        state=state,
        train_loader=loader,
        val_loader=loader,
        lm_hp=SimpleNamespace(stageB_last_hard_trig_power_max_points=4096),
        device=torch.device("cpu"),
        dtype=torch.float64,
        epochs_stageB=1,
        loss_scale=1.0,
        loss_good_enough_raw=1.0e-9,
        score_tol=0.0,
        scale_specs=[],
        scaling_by_axis={},
        trig_by_axis={},
        verbose=False,
        units_spec=units_spec,
        enforce_units=True,
    )

    hints = _stageB_phase_hints_for_atom(ctx, root)
    assert hints
    assert any(abs(float(h.observed_omega or 0.0) - harmonic_omega) < 0.15 for h in hints)

    cands = RuleLastHardTrigSquare1D().propose(ctx, root)
    assert cands
    assert any(abs(float((cand.meta or {}).get("omega", 0.0)) - arg_omega) < 0.15 for cand in cands)


if __name__ == "__main__":
    test_phase_prescan_finds_dimensionless_product_ratio()
    test_phase_hint_square_grid_keeps_sin_square_ambiguity()
    test_phase_prescan_records_explicit_omega_terms()
    test_phase_stable_int_hash_is_process_stable()
    test_phase_context_scan_finds_pb094_prefactor_context()
    test_phase_context_scan_rejects_unitful_phase_carrier()
    test_outer_inverse_trig_prescan_finds_arcsin_product_ratio()
    test_outer_inverse_trig_prescan_finds_arcsin_mixed_trig_carrier()
    test_outer_inverse_trig_prescan_finds_arcsin_affine_sin_carrier()
    test_outer_inverse_trig_prescan_finds_arcsin_affine_cos_carrier()
    test_outer_inverse_trig_prescan_rejects_unitful_target()
    test_outer_inverse_trig_prescan_rejects_unitful_mixed_trig_axis()
    test_phase_hint_direct_trig_closure_proposes_visible_ast()
    test_outer_inverse_trig_direct_closure_proposes_visible_ast()
    test_outer_inverse_trig_rational_closure_finds_pb109_shape_after_compound()
    test_stageB_overlap_walkers_accept_inverse_trig_wrappers()
    test_pretty_print_state_accepts_inverse_trig_root()
    test_phase_hint_reciprocal_trig_power_proposes_visible_ast()
    test_phase_context_direct_one_minus_cos_proposes_visible_ast()
    test_stageB_local_phase_scan_seeds_last_hard_trig_square()
