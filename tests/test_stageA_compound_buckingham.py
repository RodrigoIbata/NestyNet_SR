# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.


import torch

from nestynet_sr.sr_core.bridges import AtomNode, MulNode, PowNode, Var
from nestynet_sr.sr_core.units import (
    UnitSystem,
    UnitsSpec,
    check_compound_buckingham,
    eval_analytic_expr_dim,
)
from nestynet_sr.sr_search.search import (
    _stageA_coordinate_collapse_screen,
    _stageA_compound_buckingham_reason,
    _stageA_compound_buckingham_target_dim,
    _stageA_compound_structural_priority,
    _stageA_forced_monomial_leftover_candidates,
    _stageA_forced_monomial_loss_equivalent,
    _stageA_generate_unit_prefactor_exponents,
    _stageA_noisy_terminal_yspace_accept,
    _stageA_prefactor_peeled_raw_vars,
    _stageA_visible_buckingham_1d_prefactor_proposals_for_atom,
    _stageA_visible_prefactor_buckingham_transaction_reason,
)


def test_stageA_compound_buckingham_uses_local_atom_dimension():
    """A nested NN atom must be judged against its local output dimension.

    This is the pb011 pattern after peeling x0:

        x0 * NN[x1, x2, x3, x4]

    The useful compound z=x2*x3 has the same dimension as the local atom output
    and x1.  Checking it against the full y dimension incorrectly rejects it.
    """
    us = UnitSystem(("A", "B", "C", "D", "E"))
    x_dims = tuple(
        us.dim(v)
        for v in (
            [2, -2, 1, 0, -1],
            [-1, 0, 0, 0, 1],
            [-2, 1, 0, 0, 1],
            [1, -1, 0, 0, 0],
            [0, 0, 0, 0, 0],
        )
    )
    spec = UnitsSpec(
        unit_system=us,
        x_dims=x_dims,
        y_dim=us.dim([1, -2, 1, 0, 0]),
    )
    atom = AtomNode(kind="nn", var_idxs=(1, 2, 3, 4), tag="leaf")
    root = MulNode(Var(0), atom)
    z_expr = MulNode(Var(2), Var(3))
    z_dim = eval_analytic_expr_dim(z_expr, spec.x_dims)

    local_dim = _stageA_compound_buckingham_target_dim(root, atom, spec)
    assert local_dim == x_dims[1]

    ok_global, reason_global = check_compound_buckingham(
        atom_var_idxs=[1, 2, 3, 4],
        extra_var_idxs=[1, 4],
        z_dim=z_dim,
        x_dims=spec.x_dims,
        min_freedom=1,
        y_dim=spec.y_phi_dim,
    )
    assert not ok_global
    assert "remaining variables cannot span output dimension" in reason_global

    ok_local, reason_local = check_compound_buckingham(
        atom_var_idxs=[1, 2, 3, 4],
        extra_var_idxs=[1, 4],
        z_dim=z_dim,
        x_dims=spec.x_dims,
        min_freedom=1,
        y_dim=local_dim,
    )
    assert ok_local, reason_local


def test_stageA_noisy_terminal_gate_only_accepts_noise_limited_simplification():
    base = AtomNode(kind="nn", var_idxs=(0, 1), tag="leaf")
    cand = Var(0)

    ok, reason = _stageA_noisy_terminal_yspace_accept(
        base_ast=base,
        cand_ast=cand,
        base_y_mse=5.7e-2,
        cand_y_mse=5.32e-2,
        noise_floor_raw=5.38e-2,
        n_eff=2000,
    )
    assert ok, reason

    ok, reason = _stageA_noisy_terminal_yspace_accept(
        base_ast=base,
        cand_ast=cand,
        base_y_mse=5.7e-2,
        cand_y_mse=5.32e-2,
        noise_floor_raw=0.0,
        n_eff=2000,
    )
    assert not ok
    assert "positive y-space noise floor" in reason

    ok, reason = _stageA_noisy_terminal_yspace_accept(
        base_ast=base,
        cand_ast=cand,
        base_y_mse=4.0e-2,
        cand_y_mse=5.32e-2,
        noise_floor_raw=5.38e-2,
        n_eff=100000,
    )
    assert not ok
    assert "materially regresses" in reason


def test_stageA_visible_prefactor_transaction_rescues_bare_buckingham_reject():
    """Bare NN[z] remains illegal, but a visible P*NN[z] transaction may pass.

    This is the pb043 Buckingham shape:

        z = x0*x2/(x1*x3)       dimensionless
        P = x0^2*x1*x3          carries the local atom output dimension

    The Stage-A candidate layer may propose P*NN[z], but the underlying
    Buckingham rule should still reject the bare NN[z] child.
    """
    us = UnitSystem(("A", "B", "C"))
    x_dims = tuple(
        us.dim(v)
        for v in (
            [1, 0, 0],    # x0
            [0, 1, 0],    # x1
            [-1, 1, 1],   # x2 = -x0 + x1 + x3, so z is dimensionless
            [0, 0, 1],    # x3
        )
    )
    spec = UnitsSpec(
        unit_system=us,
        x_dims=x_dims,
        y_dim=us.dim([2, 1, 1]),
    )
    atom = AtomNode(kind="nn", var_idxs=(0, 1, 2, 3), tag="leaf")
    root = atom
    z_expr = MulNode(
        MulNode(Var(0), Var(2)),
        MulNode(PowNode(Var(1), -1), PowNode(Var(3), -1)),
    )

    bare_reason = _stageA_compound_buckingham_reason(
        current_ast=root,
        atom=atom,
        z_expr=z_expr,
        kind="monomial",
        extra_var_idxs=[],
        extra_input_asts=None,
        units_spec=spec,
        enforce_units=True,
    )
    assert bare_reason is not None
    assert "remaining variables cannot span output dimension" in bare_reason

    tx_reason = _stageA_visible_prefactor_buckingham_transaction_reason(
        current_ast=root,
        atom=atom,
        z_expr=z_expr,
        pattern=(1, -1, 1, -1),
        extra_var_idxs=[],
        extra_input_asts=None,
        prefactor_exponents=(2, 1, 0, 1),
        units_spec=spec,
        enforce_units=True,
    )
    assert tx_reason is None

    bad_tx_reason = _stageA_visible_prefactor_buckingham_transaction_reason(
        current_ast=root,
        atom=atom,
        z_expr=z_expr,
        pattern=(1, -1, 1, -1),
        extra_var_idxs=[],
        extra_input_asts=None,
        prefactor_exponents=(1, 0, 0, 0),
        units_spec=spec,
        enforce_units=True,
    )
    assert bad_tx_reason is not None
    assert "does not match local target" in bad_tx_reason


def test_stageA_generated_prefactor_complement_finds_pb043_representative():
    us = UnitSystem(("A", "B", "C"))
    x_dims = tuple(
        us.dim(v)
        for v in (
            [1, 0, 0],
            [0, 1, 0],
            [-1, 1, 1],
            [0, 0, 1],
        )
    )
    spec = UnitsSpec(
        unit_system=us,
        x_dims=x_dims,
        y_dim=us.dim([2, 1, 1]),
    )
    atom = AtomNode(kind="nn", var_idxs=(0, 1, 2, 3), tag="leaf")
    z_expr = MulNode(
        MulNode(Var(0), Var(2)),
        MulNode(PowNode(Var(1), -1), PowNode(Var(3), -1)),
    )

    pref, reason = _stageA_generate_unit_prefactor_exponents(
        current_ast=atom,
        atom=atom,
        z_expr=z_expr,
        pattern=(1, -1, 1, -1),
        extra_var_idxs=[],
        extra_input_asts=None,
        units_spec=spec,
        enforce_units=True,
    )
    assert reason == ""
    assert pref == (2, 1, 0, 1)


def test_stageA_generated_prefactor_complement_can_peel_dimensionful_extra():
    """PR3 must work before pb043's x4^-2 separability split is committed.

    The bare compound leaves x4 as a dimensionful residual extra, so it should
    only pass if the generated visible prefactor consumes x4 and the residual
    NN is left as a dimensionless function of z alone.
    """
    us = UnitSystem(("L", "T", "M", "I", "Th"))
    x_dims = tuple(
        us.dim(v)
        for v in (
            [0, -1, 0, 0, 0],   # x0
            [0, 0, 0, 1, 0],    # x1
            [2, -1, 1, 0, 0],   # x2
            [2, -2, 1, -1, 0],  # x3
            [1, -1, 0, 0, 0],   # x4
        )
    )
    spec = UnitsSpec(
        unit_system=us,
        x_dims=x_dims,
        y_dim=us.dim([0, -2, 1, 0, 0]),
    )
    atom = AtomNode(kind="nn", var_idxs=(0, 1, 2, 3, 4), tag="leaf")
    z_expr = MulNode(
        MulNode(Var(0), Var(2)),
        MulNode(PowNode(Var(1), -1), PowNode(Var(3), -1)),
    )

    pref, reason = _stageA_generate_unit_prefactor_exponents(
        current_ast=atom,
        atom=atom,
        z_expr=z_expr,
        pattern=(1, -1, 1, -1, 0),
        extra_var_idxs=[4],
        extra_input_asts=None,
        units_spec=spec,
        enforce_units=True,
    )
    assert reason == ""
    assert pref == (2, 1, 0, 1, -2)
    assert _stageA_prefactor_peeled_raw_vars(atom, pref) == {0, 1, 3, 4}

    tx_reason = _stageA_visible_prefactor_buckingham_transaction_reason(
        current_ast=atom,
        atom=atom,
        z_expr=z_expr,
        pattern=(1, -1, 1, -1, 0),
        extra_var_idxs=[4],
        extra_input_asts=None,
        prefactor_exponents=pref,
        units_spec=spec,
        enforce_units=True,
    )
    assert tx_reason is None


def test_stageA_visible_buckingham_1d_prefactor_lane_finds_pb043_after_partial_peel():
    """The protected lane should recover the post-peel pb043 dimensional collapse.

    After Stage A has already exposed ``x4^-2 * NN[x1*x3, x0, x2]``, the local
    NN target admits:

        P_local = x0^2 * (x1*x3)
        pi      = x0*x2/(x1*x3)  (or its reciprocal wrapper)
    """
    us = UnitSystem(("L", "T", "M", "I", "Th"))
    x_dims = tuple(
        us.dim(v)
        for v in (
            [0, -1, 0, 0, 0],   # x0
            [0, 0, 0, 1, 0],    # x1
            [2, -1, 1, 0, 0],   # x2
            [2, -2, 1, -1, 0],  # x3
            [1, -1, 0, 0, 0],   # x4
        )
    )
    spec = UnitsSpec(
        unit_system=us,
        x_dims=x_dims,
        y_dim=us.dim([0, -2, 1, 0, 0]),
    )
    atom = AtomNode(
        kind="nn",
        var_idxs=(0, 1, 2, 3),
        tag="leaf",
        inputs=(MulNode(Var(1), Var(3)), Var(0), Var(2)),
    )
    root = MulNode(PowNode(Var(4), -2), atom)

    class HP:
        visible_buckingham_1d_max_candidates = 4
        visible_buckingham_1d_confidence = 0.995

    proposals = _stageA_visible_buckingham_1d_prefactor_proposals_for_atom(
        current_ast=root,
        atom=atom,
        units_spec=spec,
        enforce_units=True,
        search_hp=HP(),
        x_transform_map={},
    )

    assert proposals
    patterns = {tuple(p[0]) for p in proposals}
    assert (1, -1, -1) in patterns or (-1, 1, 1) in patterns

    matched = [
        (extra, meta) for pattern, _z, _conf, extra, meta in proposals
        if tuple(pattern) in {(1, -1, -1), (-1, 1, 1)}
    ]
    assert matched
    extra, meta = matched[0]
    assert extra == []
    assert meta["visible_buckingham_1d_prefactor"] is True
    assert meta["prefactor_exponents"] == (1, 2, 0)
    assert meta["new_arity"] == 1
    assert "x0" in meta["prefactor_readable"]
    assert "x1" in meta["prefactor_readable"]
    assert "x3" in meta["prefactor_readable"]


def test_stageA_visible_buckingham_1d_prefactor_lane_canonicalizes_pi_power_gauge():
    """Do not emit shifted ``P*pi^k`` gauges that hide simple terminal leaves.

    pb056 exposed this: all of ``P*NN[x2]``, ``P*x2*NN[x2]`` and
    ``P*x2^-1*NN[x2]`` are dimensionally legal when ``x2`` is dimensionless,
    but the canonical ballot should keep the avoidable ``x2`` power inside the
    NN response.
    """
    us = UnitSystem(("A", "B", "C"))
    spec = UnitsSpec(
        unit_system=us,
        x_dims=(
            us.dim([1, 0, 0]),    # x0
            us.dim([0, 1, 0]),    # x1
            us.dim([0, 0, 0]),    # x2 dimensionless
            us.dim([0, 0, 1]),    # x3
        ),
        y_dim=us.dim([-1, 1, -3]),
    )
    atom = AtomNode(kind="nn", var_idxs=(0, 1, 2, 3), tag="leaf")

    class HP:
        visible_buckingham_1d_max_candidates = 4
        visible_buckingham_1d_confidence = 0.995

    proposals = _stageA_visible_buckingham_1d_prefactor_proposals_for_atom(
        current_ast=atom,
        atom=atom,
        units_spec=spec,
        enforce_units=True,
        search_hp=HP(),
        x_transform_map={},
    )

    x2_proposals = [
        meta for pattern, _z, _conf, _extra, meta in proposals
        if tuple(pattern) == (0, 0, 1, 0)
    ]
    assert len(x2_proposals) == 1
    meta = x2_proposals[0]
    assert meta["prefactor_exponents"] == (-1, 1, 0, -3)
    assert meta["prefactor_pi_gauge_abs"] == 0
    assert meta["prefactor_pi_gauge_canonical_exponents"] == (-1, 1, 0, -3)


def test_stageA_compound_structural_priority_prefers_canonical_buckingham_gauge():
    base = {
        "kind": "monomial",
        "visible_prefactor_transaction": True,
        "old_arity": 4,
        "new_arity": 1,
        "pattern": (0, 0, 1, 0),
    }
    canonical = dict(base, prefactor_pi_gauge_abs=0)
    shifted = dict(base, prefactor_pi_gauge_abs=1)

    assert _stageA_compound_structural_priority(canonical) < _stageA_compound_structural_priority(shifted)


def test_stageA_generated_prefactor_complement_disabled_for_dimensionless_target():
    us = UnitSystem(("A", "B", "C"))
    x_dims = tuple(
        us.dim(v)
        for v in (
            [1, 0, 0],
            [0, 1, 0],
            [-1, 1, 1],
            [0, 0, 1],
        )
    )
    spec = UnitsSpec(
        unit_system=us,
        x_dims=x_dims,
        y_dim=us.dim([0, 0, 0]),
    )
    atom = AtomNode(kind="nn", var_idxs=(0, 1, 2, 3), tag="leaf")
    z_expr = MulNode(
        MulNode(Var(0), Var(2)),
        MulNode(PowNode(Var(1), -1), PowNode(Var(3), -1)),
    )

    pref, reason = _stageA_generate_unit_prefactor_exponents(
        current_ast=atom,
        atom=atom,
        z_expr=z_expr,
        pattern=(1, -1, 1, -1),
        extra_var_idxs=[],
        extra_input_asts=None,
        units_spec=spec,
        enforce_units=True,
    )
    assert pref is None
    assert "dimensionless" in reason


def test_stageA_coordinate_collapse_screen_accepts_true_residual_coordinate():
    z = torch.linspace(0.1, 2.0, 1000, dtype=torch.float64)
    y = torch.sin(3.0 * z) + 0.2 * z
    score = _stageA_coordinate_collapse_screen([z], y, n_bins=64)
    assert score > 0.95


def test_stageA_coordinate_collapse_screen_rejects_missing_residual_coordinate():
    g = torch.Generator().manual_seed(123)
    z = torch.linspace(0.1, 2.0, 1000, dtype=torch.float64)
    hidden = torch.randn(1000, dtype=torch.float64, generator=g)
    y = hidden + 0.01 * z
    score = _stageA_coordinate_collapse_screen([z], y, n_bins=64)
    assert score < 0.15


def test_forced_monomial_regression_uses_noise_floor_not_stageA_target():
    class LM:
        chisq_tol = 1.0e-10

    class HP:
        forced_monomial_equiv_rel = 1.0e-2
        forced_monomial_equiv_noise_mult = 100.0

    ok, reason = _stageA_forced_monomial_loss_equivalent(
        forced_loss=2.5626e-4,
        reference_loss=1.8494e-6,
        lm_hp=LM(),
        loss_scale=1.0,
        search_hp=HP(),
    )
    assert not ok
    assert "material regression" in reason

    ok2, _ = _stageA_forced_monomial_loss_equivalent(
        forced_loss=1.8500e-6,
        reference_loss=1.8494e-6,
        lm_hp=LM(),
        loss_scale=1.0,
        search_hp=HP(),
    )
    assert ok2


def test_forced_monomial_leftover_scan_finds_pb111_style_ratio():
    us = UnitSystem(("A", "B"))
    x_dims = tuple(
        us.dim(v)
        for v in (
            [1, 0],   # x0
            [0, 1],   # x1
            [0, 0],   # x2 unused here
            [0, 1],   # x3 commensurate with x1
            [-1, 0],  # x4
        )
    )
    spec = UnitsSpec(unit_system=us, x_dims=x_dims, y_dim=us.dim([3, -2]))
    atom = AtomNode(kind="nn", var_idxs=(0, 1, 3, 4), tag="leaf")
    forced = MulNode(
        MulNode(PowNode(Var(0), 2), PowNode(Var(1), -3)),
        MulNode(Var(3), PowNode(Var(4), -1)),
    )

    n = 1000
    t = torch.linspace(0.0, 1.0, n, dtype=torch.float64)
    x = torch.empty((n, 5), dtype=torch.float64)
    x[:, 0] = 1.0 + 0.7 * t
    x[:, 1] = 2.0 + t
    x[:, 2] = 1.0
    x[:, 3] = 0.35 + 0.45 * t
    x[:, 4] = 1.2 + 0.3 * torch.sin(5.0 * t)

    with torch.no_grad():
        p = torch.reshape(torch.as_tensor(
            (x[:, 0] ** 2) * (x[:, 1] ** -3) * x[:, 3] * (x[:, 4] ** -1),
            dtype=torch.float64,
        ), (-1, 1))
        pi = torch.reshape(x[:, 3] / x[:, 1], (-1, 1))
        residual = 1.0 / ((1.0 - pi ** 2) ** 2)
        y = p * residual

    class HP:
        compound_variant_screen_gate = 0.15
        forced_monomial_prefactor_fallback_screen_gate = 0.50
        forced_monomial_prefactor_fallback_max_candidates = 3
        compound_screen_bins = 64

    cands = _stageA_forced_monomial_leftover_candidates(
        atom=atom,
        forced_expr=forced,
        z_expr=forced,
        units_spec=spec,
        enforce_units=True,
        x_train=x,
        y_teacher=y,
        search_hp=HP(),
    )
    assert cands
    top = cands[0][2]
    assert ("x1" in top and "x3" in top), top
