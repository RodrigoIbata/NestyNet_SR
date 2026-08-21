# SPDX-License-Identifier: MPL-2.0

from nestynet_sr.sr_core.units import UnitSystem, UnitsSpec
from nestynet_sr.sr_gs.config import GeneralizedSymmetryConfig
from nestynet_sr.sr_gs.pi_bridge import stageA_unit_torus_pi_proposals


def test_stagea_unit_torus_pi_bridge_emits_source_tagged_proposal():
    us = UnitSystem(("L", "T"))
    spec = UnitsSpec(
        unit_system=us,
        x_dims=(us.dim([1, 0]), us.dim([0, 1]), us.dim([1, -1])),
        y_dim=us.dim([1, 0]),
    )
    cfg = GeneralizedSymmetryConfig(
        enabled=True,
        mode="propose",
        unit_torus=True,
        pi_invariants=True,
        dim_policy="augment",
        pi_max_exponent=2,
        pi_max_l1=4,
        pi_max_proposals=8,
    )

    proposals, diagnostics = stageA_unit_torus_pi_proposals(cols=(0, 1, 2), units_spec=spec, cfg=cfg)

    assert proposals
    assert diagnostics
    _pattern, _z_ast, confidence, _extra, meta = proposals[0]
    assert confidence >= 0.65
    assert meta["source"] == "gs_unit_torus"
    assert meta["gs_kind"] == "pi_invariant"
    assert "pi_exponents" in meta


def test_stagea_unit_torus_pi_bridge_audits_without_proposals():
    us = UnitSystem(("L", "T"))
    spec = UnitsSpec(
        unit_system=us,
        x_dims=(us.dim([1, 0]), us.dim([0, 1]), us.dim([1, -1])),
        y_dim=us.dim([1, 0]),
    )
    cfg = GeneralizedSymmetryConfig(
        enabled=True,
        mode="propose",
        unit_torus=True,
        pi_invariants=True,
        dim_policy="audit",
        pi_max_exponent=2,
        pi_max_l1=4,
    )

    proposals, diagnostics = stageA_unit_torus_pi_proposals(cols=(0, 1, 2), units_spec=spec, cfg=cfg)

    assert proposals == []
    assert diagnostics
