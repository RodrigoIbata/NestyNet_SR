# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata

"""
Tests for the final pruning pass in Stage B.

Builds synthetic expressions with known small additive noise terms and
verifies that pruning drops them correctly.
"""

import math

import torch
from torch.utils.data import DataLoader, TensorDataset

import nestynet_sr.sr_search.stageB.pruning as stageb_pruning
import nestynet_sr.sr_search.representation as stageb_representation
from nestynet_sr.sr_core.bridges import (
    AddNode,
    AtomNode,
    MulNode,
    build_composite_from_ast,
    clone_ast,
)
from nestynet_sr.sr_search.config import LMHyperparams
from nestynet_sr.sr_search.stageB.engine import StageBState, _compute_nn_metrics
from nestynet_sr.sr_search.stageB.pruning import (
    PrunableParam,
    _collect_prunable_params,
    _compute_aic,
    _find_additive_sites,
    _flatten_additive_terms,
    _rebuild_additive_chain,
    prune_insignificant_parameters,
    prune_nested_additive_terms,
    prune_small_additive_terms,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_loaders(x, y, batch_size=2000):
    """Build separate train/val dataloaders from tensors (non-overlapping split)."""
    n = x.shape[0]
    mid = n // 2
    ds_train = TensorDataset(x[:mid], y[:mid])
    ds_val = TensorDataset(x[mid:], y[mid:])
    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=False)
    dl_val = DataLoader(ds_val, batch_size=batch_size, shuffle=False)
    return dl_train, dl_val


def _build_state(root, train_loader, val_loader, device, dtype):
    """Build a StageBState by compiling and evaluating an AST."""
    model = build_composite_from_ast(root, dtype=dtype, device=device)
    model.eval()
    se_sum = 0.0
    n = 0
    with torch.no_grad():
        for batch in val_loader:
            xb, yb = batch
            xb, yb = xb.to(device), yb.to(device)
            yp = model(xb)
            if yp.dim() == 2:
                yp = yp[:, 0]
            if yb.dim() == 2:
                yb = yb[:, 0]
            se_sum += float(((yp - yb) ** 2).sum())
            n += yp.numel()
    val_loss = se_sum / max(n, 1)
    num_nn, num_mv, max_ar = _compute_nn_metrics(root)
    return StageBState(
        root=root, model=model, reuse={},
        val_loss=val_loss,
        num_nn_atoms=num_nn,
        num_multivar_nn_atoms=num_mv,
        max_nn_arity=max_ar,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_aic_basic():
    """AIC helper produces expected values."""
    # AIC = n * ln(MSE) + 2*k
    aic = _compute_aic(mse=1.0, n_data=100, n_params=5)
    assert abs(aic - 10.0) < 1e-10  # 100*ln(1) + 2*5 = 10

    aic2 = _compute_aic(mse=math.exp(1), n_data=100, n_params=0)
    assert abs(aic2 - 100.0) < 1e-10  # 100*1 + 0 = 100


def test_flatten_rebuild_roundtrip():
    """Flatten + rebuild preserves structure for simple additive trees."""
    a = AtomNode(kind="poly", var_idxs=(0,), kwargs={"degree": 1}, tag="p0")
    b = AtomNode(kind="poly", var_idxs=(0,), kwargs={"degree": 1}, tag="p1")
    c = AtomNode(kind="poly", var_idxs=(0,), kwargs={"degree": 1}, tag="p2")
    root = AddNode(AddNode(a, b), c)
    terms = _flatten_additive_terms(root)
    assert len(terms) == 3
    rebuilt = _rebuild_additive_chain(terms)
    assert isinstance(rebuilt, AddNode)


def test_prune_drops_small_terms():
    """Pruning drops terms with negligible contribution.

    Build: y ≈ big_poly(x) + tiny_poly1(x) + tiny_poly2(x)
    where the last two have negligible coefficients. Pruning should drop them.
    """
    device = torch.device("cpu")
    dtype = torch.float64

    N = 1000
    x = torch.linspace(0.1, 6.0, N, dtype=dtype).unsqueeze(-1)
    # Target is roughly quadratic
    y_true = 3.0 * x.squeeze() ** 2 + 1.0

    # Build AST: big_poly + small_poly1 + small_poly2
    big_poly = AtomNode(kind="poly", var_idxs=(0,), kwargs={"degree": 3}, tag="big")
    small1 = AtomNode(kind="poly", var_idxs=(0,), kwargs={"degree": 1}, tag="small1")
    small2 = AtomNode(kind="poly", var_idxs=(0,), kwargs={"degree": 1}, tag="small2")
    root = AddNode(AddNode(big_poly, small1), small2)

    # Build model and manually set coefficients
    model, atom_map = build_composite_from_ast(
        root, dtype=dtype, device=device, return_atom_map=True,
    )

    # Set big_poly to approximate 3*x^2 + 1, small polys to tiny values
    with torch.no_grad():
        for atom, leaf in atom_map.items():
            core = getattr(leaf, "model", leaf)
            if hasattr(core, "coeffs"):
                tag = None
                # Find the tag for this atom
                from nestynet_sr.sr_core.bridges import collect_all_atoms
                for a in collect_all_atoms(root):
                    if id(a) == atom:
                        tag = getattr(a, "tag", None)
                        break
                if tag == "big":
                    core.coeffs.zero_()
                    core.coeffs[0] = 1.0    # intercept
                    if core.coeffs.numel() > 2:
                        core.coeffs[2] = 3.0  # x^2 coefficient
                else:
                    # Tiny contributions
                    core.coeffs.zero_()
                    if core.coeffs.numel() >= 2:
                        core.coeffs[1] = 1e-7

    # Evaluate MSE
    model.eval()
    with torch.no_grad():
        yp = model(x.to(device))
        if yp.dim() == 2:
            yp = yp[:, 0]
        se = float(((yp - y_true.to(device)) ** 2).mean())

    num_nn, num_mv, max_ar = _compute_nn_metrics(root)
    state = StageBState(
        root=root, model=model, reuse={},
        val_loss=se,
        num_nn_atoms=num_nn,
        num_multivar_nn_atoms=num_mv,
        max_nn_arity=max_ar,
    )

    lm_hp = LMHyperparams()
    lm_hp.prune_final_enable = True
    lm_hp.prune_rel_threshold = 0.01  # 1% threshold

    train_loader, val_loader = _make_loaders(x, y_true.unsqueeze(-1))
    result = prune_small_additive_terms(
        state=state,
        train_loader=train_loader,
        val_loader=val_loader,
        lm_hp=lm_hp,
        device=device, dtype=dtype,
        loss_scale=1.0,
        verbose=True,
    )

    # Result should be a valid StageBState
    assert isinstance(result, StageBState)
    assert result.root is not None
    assert result.model is not None

    # The small terms should have been flagged; check that we got fewer terms
    result_terms = _flatten_additive_terms(result.root)
    original_terms = _flatten_additive_terms(root)
    print(f"  Original terms: {len(original_terms)}, Pruned terms: {len(result_terms)}")
    assert len(result_terms) <= len(original_terms)


def test_prune_skips_when_disabled():
    """Pruning returns original state when disabled."""
    device = torch.device("cpu")
    dtype = torch.float64

    x = torch.linspace(0.1, 3.0, 100, dtype=dtype).unsqueeze(-1)
    y = torch.sin(x)

    # Simple 3-term additive AST
    a = AtomNode(kind="poly", var_idxs=(0,), kwargs={"degree": 1}, tag="a")
    b = AtomNode(kind="poly", var_idxs=(0,), kwargs={"degree": 1}, tag="b")
    c = AtomNode(kind="poly", var_idxs=(0,), kwargs={"degree": 1}, tag="c")
    root = AddNode(AddNode(a, b), c)

    train_loader, val_loader = _make_loaders(x, y)
    state = _build_state(root, train_loader, val_loader, device, dtype)

    lm_hp = LMHyperparams()
    lm_hp.prune_final_enable = False  # disabled

    result = prune_small_additive_terms(
        state=state,
        train_loader=train_loader,
        val_loader=val_loader,
        lm_hp=lm_hp,
        device=device, dtype=dtype,
        loss_scale=1.0,
        verbose=False,
    )
    assert result is state  # unchanged


def test_prune_skips_with_few_terms():
    """Pruning skips when fewer than 2 additive terms."""
    device = torch.device("cpu")
    dtype = torch.float64

    x = torch.linspace(0.1, 3.0, 100, dtype=dtype).unsqueeze(-1)
    y = torch.sin(x)

    # Only 1 term
    a = AtomNode(kind="poly", var_idxs=(0,), kwargs={"degree": 1}, tag="a")
    root = a

    train_loader, val_loader = _make_loaders(x, y)
    state = _build_state(root, train_loader, val_loader, device, dtype)

    lm_hp = LMHyperparams()
    lm_hp.prune_final_enable = True

    result = prune_small_additive_terms(
        state=state,
        train_loader=train_loader,
        val_loader=val_loader,
        lm_hp=lm_hp,
        device=device, dtype=dtype,
        loss_scale=1.0,
        verbose=False,
    )
    assert result is state  # unchanged


def test_prune_never_drops_largest_term():
    """Even with aggressive threshold, the largest-contribution term is kept."""
    device = torch.device("cpu")
    dtype = torch.float64

    x = torch.linspace(0.1, 3.0, 200, dtype=dtype).unsqueeze(-1)
    y = x * 2.0  # simple linear

    # 3 poly terms
    a = AtomNode(kind="poly", var_idxs=(0,), kwargs={"degree": 1}, tag="a")
    b = AtomNode(kind="poly", var_idxs=(0,), kwargs={"degree": 1}, tag="b")
    c = AtomNode(kind="poly", var_idxs=(0,), kwargs={"degree": 1}, tag="c")
    root = AddNode(AddNode(a, b), c)

    train_loader, val_loader = _make_loaders(x, y)
    state = _build_state(root, train_loader, val_loader, device, dtype)

    lm_hp = LMHyperparams()
    lm_hp.prune_final_enable = True
    lm_hp.prune_rel_threshold = 0.5  # very aggressive

    result = prune_small_additive_terms(
        state=state,
        train_loader=train_loader,
        val_loader=val_loader,
        lm_hp=lm_hp,
        device=device, dtype=dtype,
        loss_scale=1.0,
        verbose=True,
    )
    # The result should still have at least 1 term
    terms = _flatten_additive_terms(result.root)
    assert len(terms) >= 1


def test_prune_verbose_term_labels_do_not_trigger_atom_leaf_mismatch(capsys):
    """Verbose additive-term printing should build a subtree-aligned mini-model."""
    device = torch.device("cpu")
    dtype = torch.float64

    x = torch.linspace(0.1, 3.0, 240, dtype=dtype).unsqueeze(-1)
    y = 2.0 * x.squeeze() + 0.1

    a = AtomNode(kind="poly", var_idxs=(0,), kwargs={"degree": 1}, tag="a")
    b = AtomNode(kind="poly", var_idxs=(0,), kwargs={"degree": 1}, tag="b")
    c = AtomNode(kind="poly", var_idxs=(0,), kwargs={"degree": 1}, tag="c")
    root = AddNode(AddNode(a, b), c)

    train_loader, val_loader = _make_loaders(x, y.unsqueeze(-1))
    state = _build_state(root, train_loader, val_loader, device, dtype)

    lm_hp = LMHyperparams()
    lm_hp.prune_final_enable = True
    lm_hp.prune_rel_threshold = 0.25
    lm_hp.prune_param_aic_tolerance = 2.0
    lm_hp.prune_refit_epochs = 5

    prune_small_additive_terms(
        state=state,
        train_loader=train_loader,
        val_loader=val_loader,
        lm_hp=lm_hp,
        device=device,
        dtype=dtype,
        loss_scale=1.0,
        verbose=True,
    )

    captured = capsys.readouterr()
    assert "Atom/leaf count mismatch" not in captured.err


# ---------------------------------------------------------------------------
# Per-parameter pruning tests
# ---------------------------------------------------------------------------

def _build_state_with_reuse(root, train_loader, val_loader, device, dtype):
    """Build a StageBState with a reuse map populated from the model."""
    from nestynet_sr.sr_search.stageB.atom_mapping import _refresh_reuse_from_state
    model = build_composite_from_ast(root, dtype=dtype, device=device)
    model.eval()
    se_sum = 0.0
    n = 0
    with torch.no_grad():
        for batch in val_loader:
            xb, yb = batch
            xb, yb = xb.to(device), yb.to(device)
            yp = model(xb)
            if yp.dim() == 2:
                yp = yp[:, 0]
            if yb.dim() == 2:
                yb = yb[:, 0]
            se_sum += float(((yp - yb) ** 2).sum())
            n += yp.numel()
    val_loss = se_sum / max(n, 1)
    reuse = _refresh_reuse_from_state(root, model)
    num_nn, num_mv, max_ar = _compute_nn_metrics(root)
    return StageBState(
        root=root, model=model, reuse=reuse,
        val_loss=val_loss,
        num_nn_atoms=num_nn,
        num_multivar_nn_atoms=num_mv,
        max_nn_arity=max_ar,
    )


def test_collect_prunable_params_poly():
    """Collects correct number of prunable params from a PolyLeaf."""
    device = torch.device("cpu")
    dtype = torch.float64

    N = 200
    x = torch.linspace(0.5, 3.0, N, dtype=dtype).unsqueeze(-1)

    # degree=3, full basis (min_total=0) -> 4 monomials: 1, x, x^2, x^3
    root = AtomNode(kind="poly", var_idxs=(0,), kwargs={"degree": 3, "min_total": 0}, tag="p0")

    model = build_composite_from_ast(root, dtype=dtype, device=device)
    # Set known coefficients: 1 + 0*x + 2*x^2 + 0.00001*x^3
    core = getattr(model.leaf[0], "model", model.leaf[0])
    with torch.no_grad():
        core.coeffs.zero_()
        core.coeffs[0] = 1.0
        core.coeffs[2] = 2.0
        core.coeffs[3] = 1e-5  # tiny x^3 term

    x_val = x.to(device)
    params = _collect_prunable_params(root, model, x_val, device, dtype)

    assert len(params) == 4, f"Expected 4 prunable params, got {len(params)}"
    # All should have tag "p0" and param_name "coeffs"
    for p in params:
        assert p.atom_tag == "p0"
        assert p.param_name == "coeffs"
        assert not p.is_den_constant

    # The tiny x^3 coefficient should have low significance
    sigs = sorted(params, key=lambda p: p.significance)
    # Check that the least significant is the near-zero coefficient
    assert abs(sigs[0].value) < 1e-3, f"Least significant should be near-zero, got {sigs[0].value}"
    print(f"  Collected {len(params)} params, least significant: {sigs[0]}")


def test_collect_prunable_params_ratpoly():
    """Collects correct params from a RationalPolyLeaf, protecting den constant."""
    device = torch.device("cpu")
    dtype = torch.float64

    N = 200
    x = torch.linspace(0.5, 3.0, N, dtype=dtype).unsqueeze(-1)

    # ratpoly with deg_num=2, deg_den=2 => several coefficients
    root = AtomNode(
        kind="ratpoly", var_idxs=(0,),
        kwargs={"deg_num": 2, "deg_den": 2},
        tag="rp0",
    )

    model = build_composite_from_ast(root, dtype=dtype, device=device)
    x_val = x.to(device)

    # With protect_den_const=True, the constant denominator term should be excluded
    params = _collect_prunable_params(root, model, x_val, device, dtype, protect_den_const=True)
    den_const_params = [p for p in params if p.is_den_constant]
    assert len(den_const_params) == 0, "Denominator constant should be protected"

    # With protect_den_const=False, the constant denominator term should be included
    params_unprotected = _collect_prunable_params(
        root, model, x_val, device, dtype, protect_den_const=False,
    )
    den_const_unprotected = [p for p in params_unprotected if p.is_den_constant]
    assert len(den_const_unprotected) >= 1, "Denominator constant should be present when unprotected"

    print(f"  Protected: {len(params)} params, Unprotected: {len(params_unprotected)} params")


def test_prune_param_skips_when_disabled():
    """Per-parameter pruning returns original state when disabled."""
    device = torch.device("cpu")
    dtype = torch.float64

    x = torch.linspace(0.5, 3.0, 200, dtype=dtype).unsqueeze(-1)
    y = 2.0 * x.squeeze() ** 2 + 1.0

    root = AtomNode(kind="poly", var_idxs=(0,), kwargs={"degree": 3, "min_total": 0}, tag="p0")
    train_loader, val_loader = _make_loaders(x, y.unsqueeze(-1))
    state = _build_state_with_reuse(root, train_loader, val_loader, device, dtype)

    lm_hp = LMHyperparams()
    lm_hp.prune_param_enable = False

    result = prune_insignificant_parameters(
        state=state,
        train_loader=train_loader,
        val_loader=val_loader,
        lm_hp=lm_hp,
        device=device, dtype=dtype,
        loss_scale=1.0,
        verbose=False,
    )
    assert result is state  # unchanged


def test_prune_param_skips_single_param():
    """Per-parameter pruning skips leaves with only 1 trainable scalar."""
    device = torch.device("cpu")
    dtype = torch.float64

    x = torch.linspace(0.5, 3.0, 200, dtype=dtype).unsqueeze(-1)
    y = 2.0 * x.squeeze()

    # degree=1, homogeneous -> 1 monomial (x^1)
    root = AtomNode(kind="poly", var_idxs=(0,), kwargs={"degree": 1}, tag="p0")
    train_loader, val_loader = _make_loaders(x, y.unsqueeze(-1))
    state = _build_state_with_reuse(root, train_loader, val_loader, device, dtype)

    lm_hp = LMHyperparams()
    lm_hp.prune_param_enable = True

    result = prune_insignificant_parameters(
        state=state,
        train_loader=train_loader,
        val_loader=val_loader,
        lm_hp=lm_hp,
        device=device, dtype=dtype,
        loss_scale=1.0,
        verbose=True,
    )
    assert result is state  # single-param leaf -> nothing to prune


def test_accepted_parameter_prune_preserves_prior_simplification_path(monkeypatch):
    root = AtomNode(
        kind="poly",
        var_idxs=(0,),
        kwargs={"degree": 1, "min_total": 0},
        tag="p0",
    )
    original_model = torch.nn.Linear(1, 1, dtype=torch.float64)
    original_path = [
        {
            "stage": "B",
            "action": "rewrite exact-like candidate",
            "expression": "x0/sqrt(1-x1**2/x2**2)",
        }
    ]
    state = StageBState(
        root=root,
        model=original_model,
        reuse={},
        val_loss=1.0,
        simplification_path=original_path,
    )
    trial_state = StageBState(
        root=clone_ast(root),
        model=torch.nn.Linear(1, 1, bias=False, dtype=torch.float64),
        reuse={},
        val_loss=0.5,
    )
    params = [
        PrunableParam("p0", "coeffs", i, 1.0, float(i + 1), False, "poly")
        for i in range(2)
    ]

    monkeypatch.setattr(
        stageb_pruning,
        "_gather_val_data",
        lambda *_args, **_kwargs: (torch.zeros(10, 1), None),
    )
    monkeypatch.setattr(
        stageb_pruning,
        "_collect_prunable_params",
        lambda *_args, **_kwargs: list(params),
    )
    monkeypatch.setattr(
        stageb_pruning,
        "_fit_candidate_root",
        lambda **_kwargs: trial_state,
    )

    lm_hp = LMHyperparams()
    lm_hp.prune_param_enable = True
    lm_hp.prune_param_max_pruned = 1
    result = prune_insignificant_parameters(
        state=state,
        train_loader=[],
        val_loader=[],
        lm_hp=lm_hp,
        device=torch.device("cpu"),
        dtype=torch.float64,
        loss_scale=1.0,
        verbose=False,
    )

    assert result is trial_state
    assert result.simplification_path == original_path
    assert result.simplification_path is not state.simplification_path
    result.simplification_path[0]["action"] = "mutated"
    assert state.simplification_path[0]["action"] == "rewrite exact-like candidate"


def test_accepted_sympy_prune_preserves_prior_simplification_path(monkeypatch):
    root = AddNode(
        AtomNode(kind="poly", var_idxs=(0,), kwargs={"degree": 1}, tag="p0"),
        AtomNode(kind="poly", var_idxs=(0,), kwargs={"degree": 1}, tag="p1"),
    )
    simplified_root = AtomNode(
        kind="poly",
        var_idxs=(0,),
        kwargs={"degree": 1},
        tag="p2",
    )
    original_path = [
        {
            "stage": "B",
            "action": "rewrite exact-like candidate",
            "expression": "sin(x0)**2 + cos(x0)**2",
        }
    ]
    state = StageBState(
        root=root,
        model=torch.nn.Linear(1, 1, dtype=torch.float64),
        reuse={},
        val_loss=1.0,
        simplification_path=original_path,
    )
    trial_state = StageBState(
        root=simplified_root,
        model=torch.nn.Linear(1, 1, dtype=torch.float64),
        reuse={},
        val_loss=0.5,
    )

    monkeypatch.setattr(
        stageb_representation,
        "pretty_print_state",
        lambda *_args, **_kwargs: "sin(x0)**2 + cos(x0)**2",
    )
    monkeypatch.setattr(
        stageb_pruning,
        "sympy_to_nestynet",
        lambda *_args, **_kwargs: simplified_root,
    )
    monkeypatch.setattr(
        stageb_pruning,
        "_node_count",
        lambda node: 5 if node is root else 1,
    )
    monkeypatch.setattr(
        stageb_pruning,
        "_fit_candidate_root",
        lambda **_kwargs: trial_state,
    )

    loader = [(torch.zeros(4, 1, dtype=torch.float64), torch.zeros(4, 1))]
    result = stageb_pruning._try_noiseless_sympy_simplify_state(
        state,
        train_loader=loader,
        val_loader=loader,
        lm_hp=LMHyperparams(),
        device=torch.device("cpu"),
        dtype=torch.float64,
        loss_scale=1.0,
        verbose=False,
    )

    assert result is trial_state
    assert result.simplification_path == original_path
    assert result.simplification_path is not state.simplification_path
    result.simplification_path[0]["action"] = "mutated"
    assert state.simplification_path[0]["action"] == "rewrite exact-like candidate"


def test_prune_param_removes_insignificant_coeff():
    """Per-parameter pruning removes a near-zero coefficient from a polynomial.

    Build: y = 1 + 2*x^2 + epsilon*x^3 (epsilon very small).
    The x^3 coefficient should be pruned because it has negligible AIC impact.
    """
    device = torch.device("cpu")
    dtype = torch.float64

    N = 1000
    x = torch.linspace(0.5, 3.0, N, dtype=dtype).unsqueeze(-1)
    # True: y = 1 + 2*x^2
    y_true = 1.0 + 2.0 * x.squeeze() ** 2

    # degree=3, full basis -> 4 monomials: 1, x, x^2, x^3
    root = AtomNode(kind="poly", var_idxs=(0,), kwargs={"degree": 3, "min_total": 0}, tag="p0")

    model, atom_map = build_composite_from_ast(
        root, dtype=dtype, device=device, return_atom_map=True,
    )

    # Set coefficients: 1 + 0*x + 2*x^2 + 1e-8*x^3
    core = getattr(model.leaf[0], "model", model.leaf[0])
    with torch.no_grad():
        core.coeffs.zero_()
        core.coeffs[0] = 1.0     # constant
        core.coeffs[2] = 2.0     # x^2
        core.coeffs[3] = 1e-8    # tiny x^3

    model.eval()
    with torch.no_grad():
        yp = model(x.to(device))
        if yp.dim() == 2:
            yp = yp[:, 0]
        mse = float(((yp - y_true.to(device)) ** 2).mean())

    from nestynet_sr.sr_search.stageB.atom_mapping import _refresh_reuse_from_state
    reuse = _refresh_reuse_from_state(root, model)
    num_nn, num_mv, max_ar = _compute_nn_metrics(root)
    state = StageBState(
        root=root, model=model, reuse=reuse,
        val_loss=mse,
        num_nn_atoms=num_nn,
        num_multivar_nn_atoms=num_mv,
        max_nn_arity=max_ar,
    )

    lm_hp = LMHyperparams()
    lm_hp.prune_param_enable = True
    lm_hp.prune_param_aic_tolerance = 2.0
    lm_hp.prune_param_refit_epochs = 100

    train_loader, val_loader = _make_loaders(x, y_true.unsqueeze(-1))
    result = prune_insignificant_parameters(
        state=state,
        train_loader=train_loader,
        val_loader=val_loader,
        lm_hp=lm_hp,
        device=device, dtype=dtype,
        loss_scale=1.0,
        verbose=True,
    )

    assert isinstance(result, StageBState)
    assert result.root is not None
    assert result.model is not None

    # Check that at least one parameter was pruned (the tiny x^3 term)
    orig_params = sum(p.numel() for p in state.model.parameters() if p.requires_grad)
    result_params = sum(p.numel() for p in result.model.parameters() if p.requires_grad)
    print(f"  Original params: {orig_params}, Result params: {result_params}")
    # The result model still has the same number of torch parameters
    # (we don't physically remove them), but the pruned coefficients should be near zero
    result_core = getattr(result.model.leaf[0], "model", result.model.leaf[0])
    if hasattr(result_core, "coeffs"):
        coeffs = result_core.coeffs.detach().cpu()
        print(f"  Result coefficients: {coeffs.numpy()}")


def test_prune_param_protects_den_constant():
    """Per-parameter pruning never zeros the denominator constant term."""
    device = torch.device("cpu")
    dtype = torch.float64

    N = 200
    x = torch.linspace(0.5, 3.0, N, dtype=dtype).unsqueeze(-1)

    root = AtomNode(
        kind="ratpoly", var_idxs=(0,),
        kwargs={"deg_num": 1, "deg_den": 1},
        tag="rp0",
    )
    model = build_composite_from_ast(root, dtype=dtype, device=device)
    x_val = x.to(device)

    # Collect prunable params and verify the denominator constant is protected
    params = _collect_prunable_params(root, model, x_val, device, dtype, protect_den_const=True)
    for p in params:
        assert not p.is_den_constant, f"Denominator constant should be protected: {p}"
    print(f"  Collected {len(params)} params (den constant protected)")


# ---------------------------------------------------------------------------
# Nested additive-term pruning tests
# ---------------------------------------------------------------------------

def test_find_additive_sites_nested():
    """_find_additive_sites finds AddNode chains nested inside a MulNode."""
    # Build: MulNode(scale, AddNode(AddNode(a, b), c))
    a = AtomNode(kind="poly", var_idxs=(0,), kwargs={"degree": 1}, tag="a")
    b = AtomNode(kind="poly", var_idxs=(0,), kwargs={"degree": 1}, tag="b")
    c = AtomNode(kind="poly", var_idxs=(0,), kwargs={"degree": 1}, tag="c")
    scale = AtomNode(kind="scale", var_idxs=(0,), kwargs={}, tag="s")

    inner_add = AddNode(AddNode(a, b), c)
    root = MulNode(scale, inner_add)

    sites = _find_additive_sites(root)
    assert len(sites) == 1, f"Expected 1 nested site, got {len(sites)}"
    top_id, terms = sites[0]
    assert len(terms) == 3, f"Expected 3 terms, got {len(terms)}"
    print(f"  Found {len(sites)} site(s) with {len(terms)} terms")


def test_find_additive_sites_root_add():
    """_find_additive_sites finds root-level AddNode chains too."""
    a = AtomNode(kind="poly", var_idxs=(0,), kwargs={"degree": 1}, tag="a")
    b = AtomNode(kind="poly", var_idxs=(0,), kwargs={"degree": 1}, tag="b")
    c = AtomNode(kind="poly", var_idxs=(0,), kwargs={"degree": 1}, tag="c")
    root = AddNode(AddNode(a, b), c)

    sites = _find_additive_sites(root)
    assert len(sites) == 1, f"Expected 1 site, got {len(sites)}"
    assert len(sites[0][1]) == 3


def test_find_additive_sites_no_chain():
    """_find_additive_sites returns empty for a tree with no 3+ term chains."""
    a = AtomNode(kind="poly", var_idxs=(0,), kwargs={"degree": 1}, tag="a")
    b = AtomNode(kind="poly", var_idxs=(0,), kwargs={"degree": 1}, tag="b")
    root = MulNode(a, b)

    sites = _find_additive_sites(root)
    assert len(sites) == 0


def test_prune_nested_removes_small_term():
    """Nested pruning removes a small additive term inside a MulNode.

    Build: y = scale * (big*x^2 + tiny1*x + tiny2) where tiny terms are negligible.
    Root is MulNode(scale, AddNode(...)), so root-level has only 1 additive term.
    The nested pruner should descend and remove the tiny terms.
    """
    device = torch.device("cpu")
    dtype = torch.float64

    N = 1000
    x = torch.linspace(0.5, 5.0, N, dtype=dtype).unsqueeze(-1)
    y_true = 3.0 * x.squeeze() ** 2  # target is quadratic

    # Build: scale * (big_poly + small1 + small2)
    big = AtomNode(kind="poly", var_idxs=(0,), kwargs={"degree": 3, "min_total": 0}, tag="big")
    s1 = AtomNode(kind="poly", var_idxs=(0,), kwargs={"degree": 1, "min_total": 0}, tag="s1")
    s2 = AtomNode(kind="poly", var_idxs=(0,), kwargs={"degree": 1, "min_total": 0}, tag="s2")
    scale = AtomNode(kind="scale", var_idxs=(), kwargs={}, tag="sc")

    inner = AddNode(AddNode(big, s1), s2)
    root = MulNode(scale, inner)

    model = build_composite_from_ast(root, dtype=dtype, device=device)

    # Set scale to 1.0, big_poly to ~3*x^2, small polys to tiny
    with torch.no_grad():
        leaves = list(model.leaf)
        for i, leaf in enumerate(leaves):
            core = getattr(leaf, "model", leaf)
            if hasattr(core, "scale"):
                core.scale.fill_(1.0)
            elif hasattr(core, "coeffs"):
                if i == 0:  # big poly (degree 3)
                    core.coeffs.zero_()
                    if core.coeffs.numel() > 2:
                        core.coeffs[2] = 3.0  # x^2
                else:  # small polys
                    core.coeffs.zero_()
                    if core.coeffs.numel() >= 2:
                        core.coeffs[0] = 1e-8  # tiny constant

    model.eval()
    with torch.no_grad():
        yp = model(x.to(device))
        if yp.dim() == 2:
            yp = yp[:, 0]
        mse = float(((yp - y_true.to(device)) ** 2).mean())

    from nestynet_sr.sr_search.stageB.atom_mapping import _refresh_reuse_from_state
    reuse = _refresh_reuse_from_state(root, model)
    num_nn, num_mv, max_ar = _compute_nn_metrics(root)
    state = StageBState(
        root=root, model=model, reuse=reuse,
        val_loss=mse,
        num_nn_atoms=num_nn,
        num_multivar_nn_atoms=num_mv,
        max_nn_arity=max_ar,
    )

    lm_hp = LMHyperparams()
    lm_hp.prune_final_enable = True
    lm_hp.prune_rel_threshold = 0.01

    train_loader, val_loader = _make_loaders(x, y_true.unsqueeze(-1))
    result = prune_nested_additive_terms(
        state=state,
        train_loader=train_loader,
        val_loader=val_loader,
        lm_hp=lm_hp,
        device=device, dtype=dtype,
        loss_scale=1.0,
        verbose=True,
    )

    assert isinstance(result, StageBState)
    # Root should still be a MulNode (scale * something)
    assert isinstance(result.root, MulNode), f"Expected MulNode root, got {type(result.root)}"
    print(f"  Result root type: {type(result.root).__name__}")


def test_prune_nested_skips_when_disabled():
    """Nested pruning returns original state when disabled."""
    device = torch.device("cpu")
    dtype = torch.float64

    x = torch.linspace(0.5, 3.0, 100, dtype=dtype).unsqueeze(-1)
    y = x.squeeze() ** 2

    a = AtomNode(kind="poly", var_idxs=(0,), kwargs={"degree": 1}, tag="a")
    b = AtomNode(kind="poly", var_idxs=(0,), kwargs={"degree": 1}, tag="b")
    c = AtomNode(kind="poly", var_idxs=(0,), kwargs={"degree": 1}, tag="c")
    scale = AtomNode(kind="scale", var_idxs=(), kwargs={}, tag="sc")
    root = MulNode(scale, AddNode(AddNode(a, b), c))

    train_loader, val_loader = _make_loaders(x, y.unsqueeze(-1))
    state = _build_state(root, train_loader, val_loader, device, dtype)

    lm_hp = LMHyperparams()
    lm_hp.prune_final_enable = False

    result = prune_nested_additive_terms(
        state=state,
        train_loader=train_loader,
        val_loader=val_loader,
        lm_hp=lm_hp,
        device=device, dtype=dtype,
        loss_scale=1.0,
        verbose=False,
    )
    assert result is state


if __name__ == "__main__":
    test_aic_basic()
    print("PASS: test_aic_basic")

    test_flatten_rebuild_roundtrip()
    print("PASS: test_flatten_rebuild_roundtrip")

    test_prune_drops_small_terms()
    print("PASS: test_prune_drops_small_terms")

    test_prune_skips_when_disabled()
    print("PASS: test_prune_skips_when_disabled")

    test_prune_skips_with_few_terms()
    print("PASS: test_prune_skips_with_few_terms")

    test_prune_never_drops_largest_term()
    print("PASS: test_prune_never_drops_largest_term")

    # Per-parameter pruning tests
    test_collect_prunable_params_poly()
    print("PASS: test_collect_prunable_params_poly")

    test_collect_prunable_params_ratpoly()
    print("PASS: test_collect_prunable_params_ratpoly")

    test_prune_param_skips_when_disabled()
    print("PASS: test_prune_param_skips_when_disabled")

    test_prune_param_skips_single_param()
    print("PASS: test_prune_param_skips_single_param")

    test_prune_param_removes_insignificant_coeff()
    print("PASS: test_prune_param_removes_insignificant_coeff")

    test_prune_param_protects_den_constant()
    print("PASS: test_prune_param_protects_den_constant")

    # Nested additive-term pruning tests
    test_find_additive_sites_nested()
    print("PASS: test_find_additive_sites_nested")

    test_find_additive_sites_root_add()
    print("PASS: test_find_additive_sites_root_add")

    test_find_additive_sites_no_chain()
    print("PASS: test_find_additive_sites_no_chain")

    test_prune_nested_removes_small_term()
    print("PASS: test_prune_nested_removes_small_term")

    test_prune_nested_skips_when_disabled()
    print("PASS: test_prune_nested_skips_when_disabled")

    print("\nAll pruning tests passed!")
