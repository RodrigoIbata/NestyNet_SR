# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata

import copy
from types import SimpleNamespace

import torch

from nestynet_sr.sr_core.bridges import (
    AddNode,
    AtomNode,
    ConstNode,
    CosNode,
    ExpNode,
    LogNode,
    MulNode,
    PowNode,
    SinNode,
    Var,
    ast_to_human_readable,
    build_composite_from_ast,
)
from nestynet_sr.sr_core.units import UnitSystem, UnitsSpec, eval_analytic_expr_dim
from nestynet_sr.sr_search.ast_utils import compact_expression_repr
from nestynet_sr.sr_search.stageB.atom_mapping import _collect_univariate_nn_atoms
from nestynet_sr.sr_search.stageB.engine import (
    StageBState,
    _Checkpoint,
    _annotate_nonsense_units_leaves,
    _below_floor_regression_rejected,
    _restore_from_checkpoint,
)
from nestynet_sr.sr_search.stageB.rules import RuleNonsenseUnitsZeroPrune


def test_nonsense_units_leaf_is_tagged_displayed_and_skipped():
    us = UnitSystem(("L", "T"))
    root = AtomNode(kind="nn", var_idxs=(0,), kwargs={}, tag="leaf0")
    state = StageBState(root=root, model=torch.nn.Identity(), reuse={}, val_loss=0.0)
    spec = UnitsSpec(
        unit_system=us,
        x_dims=(us.dim({"L": 1}),),
        y_dim=us.dim({"L": 2, "T": -4}),
    )

    problems = _annotate_nonsense_units_leaves(
        state,
        units_spec=spec,
        enforce_units=True,
    )

    assert len(problems) == 1
    assert root.kwargs["_problem_label"] == "nonsense_units"
    assert ast_to_human_readable(root) == "nonsense_units[x0]"
    assert compact_expression_repr(root, use_color=False) == "nonsense_units(x0)"
    assert _collect_univariate_nn_atoms(root) == []


def test_declared_constants_prevent_false_positive_nonsense_units_leaf():
    us = UnitSystem(("L", "T"))
    root = AtomNode(kind="nn", var_idxs=(0,), kwargs={}, tag="leaf0")
    state = StageBState(root=root, model=torch.nn.Identity(), reuse={}, val_loss=0.0)
    spec = UnitsSpec(
        unit_system=us,
        x_dims=(us.dim({"L": 1}),),
        y_dim=us.dim({"L": 1, "T": -1}),
        free_const_dims={"time_scale": us.dim({"T": 1})},
    )

    problems = _annotate_nonsense_units_leaves(
        state,
        units_spec=spec,
        enforce_units=True,
    )

    assert problems == []
    assert "_problem_label" not in root.kwargs
    assert _collect_univariate_nn_atoms(root) == [root]


def test_nonsense_units_zero_prune_rule_targets_flagged_leaf():
    root = AtomNode(
        kind="nn",
        var_idxs=(0,),
        kwargs={"_problem_label": "nonsense_units"},
        tag="leaf0",
    )
    ctx = SimpleNamespace(state=SimpleNamespace(root=root))

    rule = RuleNonsenseUnitsZeroPrune()
    targets = rule.iter_targets(ctx)
    cands = rule.propose(ctx, root)

    assert targets == [root]
    assert len(cands) == 1
    assert cands[0].label == "nonsense_units_zero_prune"
    assert cands[0].root.value == 0.0


def test_nonsense_units_candidate_probe_does_not_mutate_shared_current_ast():
    us = UnitSystem(("L", "T"))
    shared_leaf = AtomNode(kind="nn", var_idxs=(0,), kwargs={}, tag="leaf0")
    current_root = MulNode(shared_leaf, Var(0))
    candidate_root = MulNode(shared_leaf, Var(0))
    cand_state = StageBState(
        root=candidate_root,
        model=torch.nn.Identity(),
        reuse={},
        val_loss=0.0,
    )
    spec = UnitsSpec(
        unit_system=us,
        x_dims=(us.dim({"L": 1}),),
        y_dim=us.dim({"L": 3, "T": -1}),
    )

    problems = _annotate_nonsense_units_leaves(
        cand_state,
        units_spec=spec,
        enforce_units=True,
        mutate=False,
    )

    assert len(problems) == 1
    assert "_problem_label" not in shared_leaf.kwargs
    assert ast_to_human_readable(current_root) == "(NN[x0] * x0)"


def test_nonsense_units_annotation_clears_stale_reachable_marker():
    us = UnitSystem(("L",))
    root = AtomNode(
        kind="nn",
        var_idxs=(0,),
        kwargs={"_problem_label": "nonsense_units", "_problem_msg": "stale"},
        tag="leaf0",
    )
    state = StageBState(root=root, model=torch.nn.Identity(), reuse={}, val_loss=0.0)
    spec = UnitsSpec(
        unit_system=us,
        x_dims=(us.dim({"L": 1}),),
        y_dim=us.dim({"L": 1}),
    )

    problems = _annotate_nonsense_units_leaves(
        state,
        units_spec=spec,
        enforce_units=True,
    )

    assert problems == []
    assert "_problem_label" not in root.kwargs
    assert ast_to_human_readable(root) == "NN[x0]"


def test_unary_dim_requires_dimensionless_argument():
    us = UnitSystem(("L", "T"))
    x_dims = (
        us.dim({"L": 1}),
        us.dim({"L": 1}),
        us.dim({}),
    )

    dimless_ratio = MulNode(Var(0), PowNode(Var(1), -1.0))

    assert eval_analytic_expr_dim(CosNode(dimless_ratio), x_dims) == us.dim({})
    assert eval_analytic_expr_dim(SinNode(Var(2)), x_dims) == us.dim({})
    assert eval_analytic_expr_dim(LogNode(dimless_ratio), x_dims) == us.dim({})
    assert eval_analytic_expr_dim(ExpNode(dimless_ratio), x_dims) == us.dim({})
    assert eval_analytic_expr_dim(CosNode(Var(0)), x_dims) is None
    assert eval_analytic_expr_dim(LogNode(Var(0)), x_dims) is None
    assert eval_analytic_expr_dim(LogNode(MulNode(Var(0), Var(2))), x_dims) is None
    assert eval_analytic_expr_dim(AddNode(Var(0), Var(2)), x_dims) is None


def test_trig_full_compound_inputs_prevent_false_nonsense_units_tags():
    us = UnitSystem(("L", "T", "M", "I", "Theta"))
    x_dims = (
        us.dim({"L": 1, "T": -2, "M": 1, "Theta": -2}),
        us.dim({"L": 3, "T": -2, "M": 1, "Theta": -1}),
        us.dim({}),
        us.dim({"L": 1}),
    )
    z_expr = MulNode(
        MulNode(
            MulNode(PowNode(Var(0), -1.0), Var(1)),
            PowNode(Var(3), -2.0),
        ),
        CosNode(MulNode(ConstNode(1.0), Var(2))),
    )
    root = AtomNode(
        kind="nn",
        var_idxs=(0, 1, 2, 3),
        inputs=(z_expr,),
        kwargs={},
        tag="leaf0",
    )
    state = StageBState(root=root, model=torch.nn.Identity(), reuse={}, val_loss=0.0)
    spec = UnitsSpec(
        unit_system=us,
        x_dims=x_dims,
        y_dim=us.dim({"Theta": 1}),
    )

    problems = _annotate_nonsense_units_leaves(
        state,
        units_spec=spec,
        enforce_units=True,
    )

    assert problems == []
    assert "_problem_label" not in root.kwargs
    assert _collect_univariate_nn_atoms(root) == [root]
    assert eval_analytic_expr_dim(z_expr, x_dims) == spec.y_dim

    z_expr_sin = MulNode(
        MulNode(
            MulNode(PowNode(Var(0), -1.0), Var(1)),
            PowNode(Var(3), -3.0),
        ),
        SinNode(MulNode(ConstNode(2.0), Var(2))),
    )
    root_sin = AtomNode(
        kind="nn",
        var_idxs=(0, 1, 2, 3),
        inputs=(z_expr_sin,),
        kwargs={},
        tag="leaf1",
    )
    state_sin = StageBState(
        root=root_sin,
        model=torch.nn.Identity(),
        reuse={},
        val_loss=0.0,
    )
    spec_sin = UnitsSpec(
        unit_system=us,
        x_dims=x_dims,
        y_dim=us.dim({"L": -1, "Theta": 1}),
    )

    problems_sin = _annotate_nonsense_units_leaves(
        state_sin,
        units_spec=spec_sin,
        enforce_units=True,
    )

    assert problems_sin == []
    assert "_problem_label" not in root_sin.kwargs
    assert _collect_univariate_nn_atoms(root_sin) == [root_sin]
    assert eval_analytic_expr_dim(z_expr_sin, x_dims) == spec_sin.y_dim


def test_checkpoint_restore_preserves_simplification_path():
    root = AtomNode(kind="poly", var_idxs=(0,), kwargs={"degree": 1}, tag="poly0")
    model, _ = build_composite_from_ast(
        root,
        dtype=torch.float64,
        device=torch.device("cpu"),
        return_atom_map=True,
    )
    path = [{"step": 0, "stage": "B", "action": "rewrite monomial"}]
    ckpt = _Checkpoint(
        root=copy.deepcopy(root),
        val_loss=1.0e-12,
        model_state_dict={k: v.cpu().clone() for k, v in model.state_dict().items()},
        reuse_state_dicts=None,
        enabled_patterns=["monomial_deg1"],
        best_val_loss=1.0e-12,
        has_structural=False,
        decision_log_len=0,
        decision_step=0,
        attempted_transformations={},
        accept_step=1,
        accept_rule="univariate_mono",
        accept_label="monomial_deg1",
        accept_target="nn#leaf0",
        n_params=model.num_parameters() if hasattr(model, "num_parameters") else 0,
        simplification_path=path,
    )
    ctx = SimpleNamespace(
        device=torch.device("cpu"),
        dtype=torch.float64,
        fresh_nn_factory=None,
        lm_hp=SimpleNamespace(fit_y_link=None, fit_y_link_scale=1.0),
        atom_factory=None,
    )

    restored = _restore_from_checkpoint(ctx, ckpt)

    assert restored.simplification_path == path
    assert restored.simplification_path is not path


def test_problem_prune_bypasses_below_floor_regression_cap():
    rejected = _below_floor_regression_rejected(
        cand_loss=1.0e-7,
        below_floor_regress_cap=1.0e-9,
        is_separability_rewrite=False,
        relaxed_below_floor=True,
    )
    assert rejected is False
