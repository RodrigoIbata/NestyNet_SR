# SPDX-License-Identifier: MPL-2.0

from types import SimpleNamespace

from nestynet_sr.sr_gs.config import GeneralizedSymmetryConfig
from nestynet_sr.sr_gs.pi_bridge import stageA_unit_torus_pi_proposals


def test_unit_torus_stagea_records_ast_simplify_metadata_when_enabled():
    units_spec = SimpleNamespace(x_dims=((1,), (-1,)))
    cfg = GeneralizedSymmetryConfig(
        enabled=True,
        mode="propose",
        unit_torus=True,
        pi_invariants=True,
        dim_policy="augment",
        pi_max_exponent=1,
        pi_max_l1=2,
        ast_simplify=True,
    )
    proposals, diagnostics = stageA_unit_torus_pi_proposals(cols=(0, 1), units_spec=units_spec, cfg=cfg)
    assert diagnostics
    assert proposals
    _pattern, _z_ast, _confidence, _extra, meta = proposals[0]
    assert "ast_simplify" in meta
    assert meta["ast_simplify"]["enabled"] is True
