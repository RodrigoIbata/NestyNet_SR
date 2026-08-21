import numpy as np
import pytest
import torch

from nestynet_sr.sr_core.bridges import U
from nestynet_sr.sr_gs.jet_bundle import JetSampleTable, JetSpaceSpec
from nestynet_sr.sr_gs.prolongation import (
    point_prolongation_support,
    require_point_prolongation_support,
    score_affine_point_generators_from_jet_space,
)


def _sample_columns(names, n=7):
    base = np.linspace(0.1, 1.0, n)
    return {name: base + 0.25 * i for i, name in enumerate(names)}


def test_vector_pde_jet_space_materializes_multi_component_samples():
    jet_space = JetSpaceSpec(independent=("t", "x"), dependent=("rho", "m"), max_order=2)

    assert jet_space.jet_scope == "vector_pde"
    assert jet_space.coordinate_names(max_order=1) == ("t", "x", "rho", "rho_t", "rho_x", "m", "m_t", "m_x")
    assert "rho_tx" in jet_space.coordinate_names()
    assert "m_xx" in jet_space.coordinate_names()

    table = jet_space.materialize_jet_samples(_sample_columns(jet_space.coordinate_names()), order=1)

    assert isinstance(table, JetSampleTable)
    assert table.num_samples == 7
    assert table.coordinate_names == ("t", "x", "rho", "rho_t", "rho_x", "m", "m_t", "m_x")
    assert table.as_matrix().shape == (7, 8)
    torch.testing.assert_close(table.derivative_tensor("m", (0, 1)).reshape(-1), table.tensor("m_x").reshape(-1))
    report = table.to_report()
    assert report["jet_scope"] == "vector_pde"
    assert report["num_coordinates"] == 8


def test_vector_ode_scope_and_coordinate_names_are_first_class():
    jet_space = JetSpaceSpec(independent=("t",), dependent=("x", "y"), max_order=1)

    assert jet_space.jet_scope == "vector_ode"
    assert jet_space.is_vector_system
    assert not jet_space.is_pde
    assert jet_space.coordinate_names() == ("t", "x", "x_t", "y", "y_t")
    assert jet_space.to_report()["jet_scope"] == "vector_ode"


def test_general_jet_table_rejects_missing_and_unknown_coordinates():
    jet_space = JetSpaceSpec(independent=("t", "x"), dependent=("u",), max_order=1)

    with pytest.raises(KeyError, match="u_x"):
        jet_space.materialize_jet_samples({"t": [0.0], "x": [0.0], "u": [1.0], "u_t": [0.0]})

    with pytest.raises(KeyError, match="unknown jet coordinate"):
        jet_space.materialize_jet_samples(
            {"t": [0.0], "x": [0.0], "u": [1.0], "u_t": [0.0], "u_x": [0.0], "u_y": [0.0]},
            names=("t", "x", "u", "u_t", "u_x", "u_y"),
        )


def test_vector_pde_prolongation_support_fails_explicitly_without_overclaiming():
    jet_space = JetSpaceSpec(independent=("t", "x"), dependent=("u", "v"), max_order=2)

    status = point_prolongation_support(jet_space)

    assert not status.supported
    assert status.jet_scope == "vector_pde"
    assert "vector/PDE prolongation is not implemented" in status.reason
    assert status.to_report()["supported"] is False
    with pytest.raises(NotImplementedError, match="jet_scope=vector_pde"):
        require_point_prolongation_support(jet_space)


def test_scalar_jet_space_wrapper_preserves_existing_prolongation_scoring():
    x = torch.linspace(0.1, 2.0, 128, dtype=torch.float64).unsqueeze(1)
    u = torch.exp(-x)
    jet_space = JetSpaceSpec(independent=("x",), dependent=("u",), max_order=1)

    meta = score_affine_point_generators_from_jet_space(
        jet_space=jet_space,
        samples={"x": x, "u": u, "u_x": -u},
        term_asts=[U()],
        coeffs=torch.tensor([1.0], dtype=torch.float64),
        include_known=True,
        include_general_affine=False,
        tol=1.0e-8,
    )

    assert meta["status"] == "scored"
    assert meta["prolongation_support"]["supported"]
    assert meta["prolongation_scope"] == "scalar_ode_phase_one"
    assert "u_scaling" in meta["accepted_generator_names"]
