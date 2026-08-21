import numpy as np
import torch

from nestynet_sr.sr_core.bridges import AddNode, Var, eval_input_expr
from nestynet_sr.sr_core.carrier_units import CARRIER_INTERNAL_UNITS_INVALID
from nestynet_sr.sr_core.units import UnitSystem, UnitsSpec
from nestynet_sr.sr_gs import (
    GeneralizedSymmetryConfig,
    discover_gs_carriers,
)
from nestynet_sr.sr_gs.stagea_bridge import stageA_generalized_symmetry_proposals
from nestynet_sr.sr_search.factorized_search.expr_ast import eval_node
from nestynet_sr.sr_search.factorized_search.gs_carrier_seed import (
    discover_gs_carrier_seeds,
)
from nestynet_sr.sr_search.search import _stageA_schedule_gs_compound_lanes


def _cfg(*, max_depth=2, beam_width=2):
    return GeneralizedSymmetryConfig(
        enabled=True,
        mode="propose",
        policy="augment",
        general_affine=False,
        pairwise_composition=True,
        recursive_composition=True,
        recursive_composition_max_depth=max_depth,
        recursive_composition_beam_width=beam_width,
        max_stagea_proposals=16,
    )


def _aif006(n=600, seed=6):
    rng = np.random.default_rng(seed)
    x = rng.uniform(1.0, 5.0, size=(n, 6))
    y = x[:, 0] * x[:, 3] + x[:, 1] * x[:, 4] + x[:, 2] * x[:, 5]
    grad = np.column_stack(
        (x[:, 3], x[:, 4], x[:, 5], x[:, 0], x[:, 1], x[:, 2])
    )
    return x, y, grad


def _aif006_with_surrogate_gradient_noise(n=600, seed=6):
    """AIF006 with tiny, pair-correlated teacher-gradient errors.

    Each primitive product remains an essentially exact carrier, while the
    directional derivatives of the three virtual products differ at the
    roughly 1e-4 level measured in the saved pb006 Stage-A teacher.
    """

    x, y, _ = _aif006(n=n, seed=seed)
    phase = np.linspace(0.0, 8.0 * np.pi, n)
    eps = 1.5e-4
    virtual_derivs = np.column_stack(
        (
            1.0 + eps * np.sin(phase),
            1.0 + eps * np.cos(1.7 * phase),
            1.0 + eps * np.sin(2.3 * phase + 0.4),
        )
    )
    grad = np.empty_like(x)
    for k, (i, j) in enumerate(((0, 3), (1, 4), (2, 5))):
        grad[:, i] = x[:, j] * virtual_derivs[:, k]
        grad[:, j] = x[:, i] * virtual_derivs[:, k]
    return x, y, grad


def _values(ast, x):
    return (
        eval_input_expr(ast, torch.as_tensor(x, dtype=torch.float64))
        .detach()
        .cpu()
        .numpy()
        .reshape(-1)
    )


def test_shared_carrier_bank_jointly_composes_aif006_dot_product():
    x, y, grad = _aif006()
    carriers, diagnostics = discover_gs_carriers(
        atom=None,
        leaf=None,
        x_vals=x,
        y_vals=y,
        dydx_vals=grad,
        cols=tuple(range(6)),
        cfg=_cfg(),
    )

    full = [carrier for carrier in carriers if carrier.full_support and carrier.depth == 2]
    assert full, "three certified pair products should compose into one full carrier"
    best = full[0]
    assert best.certified
    assert len(best.parent_fingerprints) == 3
    assert abs(np.corrcoef(_values(best.ast, x), y)[0, 1]) > 1.0 - 1.0e-10
    assert any(
        row.get("route") == "joint_virtual" and row.get("accepted")
        for row in diagnostics
    )


def test_shared_carrier_bank_composes_aif006_with_surrogate_gradient_noise():
    x, y, grad = _aif006_with_surrogate_gradient_noise()
    carriers, diagnostics = discover_gs_carriers(
        atom=None,
        leaf=None,
        x_vals=x,
        y_vals=y,
        dydx_vals=grad,
        cols=tuple(range(6)),
        cfg=_cfg(),
    )

    full = [carrier for carrier in carriers if carrier.full_support and carrier.depth == 2]
    assert full, "recursive composition should tolerate trained-teacher gradient error"
    assert abs(np.corrcoef(_values(full[0].ast, x), y)[0, 1]) > 1.0 - 1.0e-10
    assert any(
        row.get("route") == "joint_virtual" and row.get("accepted")
        for row in diagnostics
    )


def test_rejected_recursive_joint_gate_retains_calibrated_diagnostics():
    x, y, grad = _aif006_with_surrogate_gradient_noise()
    cfg = _cfg()
    cfg.noise_calibrated_snap_factor = 0.1
    carriers, diagnostics = discover_gs_carriers(
        atom=None,
        leaf=None,
        x_vals=x,
        y_vals=y,
        dydx_vals=grad,
        cols=tuple(range(6)),
        cfg=cfg,
    )

    assert not any(carrier.full_support and carrier.depth == 2 for carrier in carriers)
    rejected = [
        row
        for row in diagnostics
        if row.get("kind") == "recursive_composition"
        and row.get("route") == "joint_virtual"
        and row.get("reason") == "joint_ray_residual_exceeds_tol"
        and len(row.get("inner", ())) == 3
    ]
    assert rejected
    row = rejected[0]
    assert row["pair_residual_metric"] == "joint_normalized"
    assert row["reported_max_pair_residual"] < row["baseline_pair_residual"]
    assert row["joint_residual_rel"] > row["joint_residual_tol"]


def test_stagea_adapter_exposes_the_same_canonical_recursive_carrier():
    x, y, grad = _aif006()
    carriers, _ = discover_gs_carriers(
        atom=None,
        leaf=None,
        x_vals=x,
        y_vals=y,
        dydx_vals=grad,
        cols=tuple(range(6)),
        cfg=_cfg(),
    )
    proposals, _ = stageA_generalized_symmetry_proposals(
        atom=None,
        leaf=None,
        x_vals=x,
        y_vals=y,
        dydx_vals=grad,
        cols=tuple(range(6)),
        cfg=_cfg(),
    )

    bank_keys = {carrier.fingerprint for carrier in carriers}
    proposal_keys = {
        str(proposal[4].get("gs_carrier_fingerprint"))
        for proposal in proposals
    }
    assert proposal_keys == bank_keys
    full = [proposal for proposal in proposals if all(proposal[0])]
    assert full
    assert full[0][4]["candidate_role"] == "inner_coordinate"
    assert full[0][4]["carrier_certified"] is True
    decisive, ordinary, fallback = _stageA_schedule_gs_compound_lanes(
        proposals,
        max_ordinary_proposals=6,
        max_gs_proposals=6,
        decisive_min_confidence=0.995,
        decisive_max_trials=1,
    )
    assert ordinary == []
    assert decisive
    assert (
        abs(np.corrcoef(_values(decisive[0][1], x), y)[0, 1])
        > 1.0 - 1.0e-10
    )
    assert len(decisive) + len(fallback) <= 6


def test_fss_adapter_receives_recursive_carrier_and_original_certificate():
    x, y, _grad = _aif006()
    x_tensor = torch.as_tensor(x, dtype=torch.float64)

    def target(values):
        return (
            values[:, 0] * values[:, 3]
            + values[:, 1] * values[:, 4]
            + values[:, 2] * values[:, 5]
        ).reshape(-1, 1)

    seeds, diagnostics = discover_gs_carrier_seeds(
        target,
        x_tensor,
        n_var=6,
        cfg=_cfg(),
        max_seeds=16,
    )

    matches = []
    for seed, row in zip(seeds, diagnostics):
        values = eval_node(seed, x_tensor).detach().cpu().numpy().reshape(-1)
        if abs(np.corrcoef(values, y)[0, 1]) > 1.0 - 1.0e-10:
            matches.append(row)
    assert matches
    row = matches[0]
    assert row["gs_carrier_depth"] == 2
    assert len(row["gs_recursive_parent_fingerprints"]) == 3
    assert row["carrier_metadata"]["carrier_certified"] is True
    assert row["carrier_metadata"]["candidate_role"] == "inner_coordinate"


def test_depth_three_is_bounded_and_deduplicates_equivalent_full_carriers():
    rng = np.random.default_rng(7)
    x = rng.uniform(0.7, 3.0, size=(700, 4))
    inner = x[:, 0] * x[:, 1] + x[:, 2]
    z = inner * x[:, 3]
    y = np.sin(z) + 0.1 * z**2
    outer_grad = np.cos(z) + 0.2 * z
    grad = np.column_stack(
        (
            outer_grad * x[:, 1] * x[:, 3],
            outer_grad * x[:, 0] * x[:, 3],
            outer_grad * x[:, 3],
            outer_grad * inner,
        )
    )

    carriers, _ = discover_gs_carriers(
        atom=None,
        leaf=None,
        x_vals=x,
        y_vals=y,
        dydx_vals=grad,
        cols=(0, 1, 2, 3),
        cfg=_cfg(max_depth=3, beam_width=3),
    )

    assert carriers
    assert max(carrier.depth for carrier in carriers) <= 3
    assert len({carrier.fingerprint for carrier in carriers}) == len(carriers)
    matching = [
        carrier
        for carrier in carriers
        if carrier.full_support
        and abs(np.corrcoef(_values(carrier.ast, x), z)[0, 1]) > 1.0 - 1.0e-8
    ]
    assert matching
    assert len({carrier.fingerprint for carrier in matching}) == 1


def test_shared_bank_rejects_an_internally_unit_illegal_primitive(monkeypatch):
    from nestynet_sr.sr_gs import stagea_bridge

    illegal = (
        (1, 1),
        AddNode(Var(0), Var(1)),
        1.0,
        None,
        {
            "kind": "gs_promoted_reduction",
            "source": "generalized_symmetry",
        },
    )
    monkeypatch.setattr(
        stagea_bridge,
        "_discover_generalized_symmetry_proposal_tuples",
        lambda **_kwargs: ([illegal], []),
    )
    unit_system = UnitSystem(("L", "T"))
    units_spec = UnitsSpec(
        unit_system=unit_system,
        x_dims=(
            unit_system.dim([1, 0]),
            unit_system.dim([0, 1]),
        ),
        y_dim=unit_system.dim([1, 0]),
    )
    x = np.ones((32, 2))
    carriers, diagnostics = discover_gs_carriers(
        atom=None,
        leaf=None,
        x_vals=x,
        y_vals=np.ones(32),
        dydx_vals=np.ones_like(x),
        cols=(0, 1),
        cfg=_cfg(),
        units_spec=units_spec,
    )

    assert carriers == []
    assert diagnostics[-1]["reason"] == CARRIER_INTERNAL_UNITS_INVALID
