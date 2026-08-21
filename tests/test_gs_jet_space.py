import numpy as np
import pytest
import torch

from nestynet_sr.sr_gs.jet_bundle import JetSpaceSpec


def test_jet_space_enumerates_scalar_ode_coordinates_and_reports_multi_indices():
    jet_space = JetSpaceSpec(independent=("x",), dependent=("u",), max_order=2)

    assert jet_space.coordinate_names() == ("x", "u", "u_x", "u_xx")
    assert jet_space.scalar_ode_coordinate_names(order=1) == ("x", "u", "u_x")
    assert jet_space.scalar_ode_coordinate_names(order=2) == ("x", "u", "u_x", "u_xx")

    report = jet_space.to_report()
    derivative_rows = [row for row in report["coordinates"] if row["kind"] == "derivative"]
    assert derivative_rows == [
        {"name": "u_x", "kind": "derivative", "component": "u", "multi_index": [1], "order": 1, "provenance": {}},
        {"name": "u_xx", "kind": "derivative", "component": "u", "multi_index": [2], "order": 2, "provenance": {}},
    ]


def test_jet_space_materializes_existing_scalar_prolongation_inputs():
    x = np.linspace(0.1, 1.0, 8)
    samples = {
        "x": x,
        "u": np.sin(x),
        "u_x": np.cos(x),
        "u_xx": -np.sin(x),
    }
    jet_space = JetSpaceSpec(independent=("x",), dependent=("u",), max_order=2)

    materialized = jet_space.materialize_scalar_ode_inputs(samples, order=2)
    kwargs = materialized.as_kwargs()

    assert kwargs["order"] == 2
    assert kwargs["x_axis"] == 0
    assert kwargs["x"].shape == (8, 1)
    assert kwargs["u"].shape == (8, 1)
    assert kwargs["u1"].shape == (8, 1)
    assert kwargs["u2"].shape == (8, 1)
    assert kwargs["x"].dtype == torch.float64
    torch.testing.assert_close(kwargs["u1"].reshape(-1), torch.as_tensor(np.cos(x), dtype=torch.float64))


def test_jet_space_rejects_missing_or_mismatched_scalar_samples():
    jet_space = JetSpaceSpec(independent=("x",), dependent=("u",), max_order=1)

    with pytest.raises(KeyError, match="u_x"):
        jet_space.materialize_scalar_ode_inputs({"x": [0.0], "u": [1.0]}, order=1)

    with pytest.raises(ValueError, match="share row count"):
        jet_space.materialize_scalar_ode_inputs({"x": [0.0, 1.0], "u": [1.0], "u_x": [0.0]}, order=1)


def test_jet_space_represents_pde_derivative_coordinates_without_enabling_prolongation():
    jet_space = JetSpaceSpec(independent=("t", "x"), dependent=("u",), max_order=2)

    assert jet_space.multi_indices(include_zero=True) == ((0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2))
    assert jet_space.derivative_name("u", (1, 1)) == "u_tx"
    assert "u_xx" in jet_space.coordinate_names()
    with pytest.raises(NotImplementedError, match="vector/PDE prolongation"):
        jet_space.require_scalar_ode_phase_one()
