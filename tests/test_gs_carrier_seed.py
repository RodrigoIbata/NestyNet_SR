"""SR-side GS -> FSS carrier-seed bridge.

The GS layer discovers the internal coordinate z(x); the FSS outer-map battery
then fits g(z) directly. These tests lock (a) the integer-power tuple-AST
conversion the warp coordinates need, (b) the driver discovering a coordinate
and handing it over, and (c) the end-to-end oracle_lab solve that FSS-alone
cannot reach.
"""

import numpy as np
import torch

from nestynet_sr.sr_core.bridges import AddNode, PowNode, Var
from nestynet_sr.sr_search.factorized_search.bridge import (
    nestynet_to_factorized_search as n2f,
)
from nestynet_sr.sr_search.factorized_search.gs_carrier_seed import (
    discover_gs_carrier_seeds,
)


def test_integer_power_bridge_conversion():
    # exponentiation by squaring, previously an unsupported ValueError
    assert n2f(PowNode(Var(0), 3.0)) == ("mul", ("var", 0), ("sqr", ("var", 0)))
    assert n2f(PowNode(Var(0), 4.0)) == ("sqr", ("sqr", ("var", 0)))
    neg = n2f(PowNode(Var(0), -3.0))
    assert neg[0] == "div" and neg[1] == ("const", 1.0)
    # a mixed-power sum (warp coordinate) now converts end-to-end
    tup = n2f(AddNode(AddNode(PowNode(Var(0), 2.0), PowNode(Var(1), 3.0)), PowNode(Var(2), 2.0)))
    assert tup[0] == "add"
    # exponents beyond the supported window still reject
    try:
        n2f(PowNode(Var(0), 9.0))
        raised = False
    except ValueError:
        raised = True
    assert raised


def _minkowski_leaf(Z: torch.Tensor) -> torch.Tensor:
    z = Z[:, 0] ** 2 - Z[:, 1] ** 2 - Z[:, 2] ** 2 - Z[:, 3] ** 2
    return torch.sin(z).reshape(-1, 1)


def test_driver_discovers_minkowski_coordinate():
    rng = np.random.default_rng(0)
    X = rng.uniform(0.5, 2.5, size=(500, 4))
    X[:, 1:] = rng.uniform(0.5, 1.5, size=(500, 3))
    Xt = torch.as_tensor(X, dtype=torch.float64)
    seeds, diag = discover_gs_carrier_seeds(_minkowski_leaf, Xt, n_var=4)
    assert seeds, "expected at least one GS carrier seed"
    # the Minkowski interval (four squared terms, mixed signs) is among them
    humans = " | ".join(str(d.get("z_human", "")) for d in diag)
    assert humans.count("**2") >= 4


def test_oracle_gs_carrier_seed_solves_minkowski_end_to_end():
    import dataclasses

    from nestynet_sr.sr_search.factorized_search.oracle_lab import (
        default_oracle_hyperparams,
        equation_spec_from_dict,
        run_oracle_equation,
    )

    spec = equation_spec_from_dict({
        "id": "minkowski_test",
        "basis": ["L", "T", "M"],
        "variables": [
            {"name": "x0", "bounds": [0.5, 2.5], "dim": [0, 0, 0]},
            {"name": "x1", "bounds": [0.5, 1.5], "dim": [0, 0, 0]},
            {"name": "x2", "bounds": [0.5, 1.5], "dim": [0, 0, 0]},
            {"name": "x3", "bounds": [0.5, 1.5], "dim": [0, 0, 0]},
        ],
        "constants": [],
        "target": {"expr": "sin(x0**2 - x1**2 - x2**2 - x3**2)", "dim": [0, 0, 0]},
    })
    hp = default_oracle_hyperparams()
    hp = dataclasses.replace(hp, n_iter=60, n_fit=400, n_probe=400, n_seeds=1)

    report = run_oracle_equation(
        spec, factorized_search_hp=hp, dtype=torch.float64,
        enforce_dims=False, verbose=False, gs_carrier_seed=True,
    )
    best = report.get("best")
    assert best is not None
    # the GS seed is scored at iteration 0, so the interval is recovered
    assert float(best["mse"]) < 1e-6, f"expected solve, got mse={best['mse']}"
