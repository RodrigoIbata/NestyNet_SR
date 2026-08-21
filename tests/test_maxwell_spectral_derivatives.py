import importlib.util
import sys
from pathlib import Path

import torch

MAXWELL_DIR = Path(__file__).resolve().parents[1] / "examples" / "Maxwell"


def _load_maxwell_module(name):
    """Import an examples/Maxwell module under a private, unambiguous name.

    Four examples/ directories ship a module called ``problem_defs``. A bare
    ``import problem_defs`` after ``sys.path.insert`` binds whichever test
    imported it first, for the whole session, so the winner depends on
    collection order. Loading by explicit path is order-independent and keeps
    this test from claiming the global ``problem_defs`` name.
    """
    path = MAXWELL_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_maxwell_{name}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_problem_defs = _load_maxwell_module("problem_defs")
_spectral_derivatives = _load_maxwell_module("spectral_derivatives")

PROBLEM_REGISTRY = _problem_defs.PROBLEM_REGISTRY
build_problem_data = _problem_defs.build_problem_data
build_derivative_targets = _spectral_derivatives.build_derivative_targets


def test_mw002_spectral_spatial_exact_time_targets_match_generator():
    problem = PROBLEM_REGISTRY["mw002"]
    X, Y, G, _meta = build_problem_data(problem, fast=False)
    target = build_derivative_targets("spectral_spatial_exact_time", X, Y, G)

    spatial = target[:, :, list(problem.spatial_axes)] - G[:, :, list(problem.spatial_axes)]
    time = target[:, :, 0] - G[:, :, 0]
    assert float(torch.max(torch.abs(spatial)).item()) < 1e-10
    assert torch.equal(time, torch.zeros_like(time))


def test_exact_derivative_targets_are_copied():
    problem = PROBLEM_REGISTRY["mw002"]
    X, Y, G, _meta = build_problem_data(problem, fast=True)
    target = build_derivative_targets("exact", X, Y, G)
    assert target is not G
    assert torch.equal(target, G)
