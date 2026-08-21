# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata

from __future__ import annotations

from nestynet_sr.sr_de.engines import DEEngineOutput, merge_engine_proposal_slates
from nestynet_sr.sr_de.proposals import build_proposal_slate


def _slate(equation: str):
    return build_proposal_slate(
        first_line={
            "engine": "stlsq",
            "kind": "library",
            "order": 1,
            "x_axis": 0,
            "canonical_equation": equation,
            "probe_rms": 1.0e-4,
        }
    )


def test_engine_outputs_merge_into_namespaced_proposal_reservoir():
    out0 = DEEngineOutput(engine="sparse", proposal_slate=_slate("u_x + u = 0"))
    out1 = DEEngineOutput(engine="typed", proposal_slate=_slate("-u_x - u = 0"))

    merged = merge_engine_proposal_slates([out0, out1])

    assert len(merged) == 1
    assert merged[0]["support"]["support_count"] == 2
    assert merged[0]["support"]["sources"] == ["sparse:first_line", "typed:first_line"]
    assert out0.to_dict()["engine"] == "sparse"
