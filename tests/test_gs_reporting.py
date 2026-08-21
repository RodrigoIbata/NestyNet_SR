import json
import numpy as np

from nestynet_sr.sr_gs import GeneralizedSymmetryConfig
from nestynet_sr.sr_gs.reporting import reset_gs_reporter, write_gs_reports
from nestynet_sr.sr_gs.stagea_bridge import stageA_generalized_symmetry_proposals


class RatioLeaf:
    def __call__(self, x):
        return x[:, 0:1] / x[:, 1:2]


def test_gs_report_contains_satisfied_and_switched_off_generators(tmp_path):
    rng = np.random.default_rng(42)
    X = rng.uniform(0.5, 2.0, size=(128, 2))
    G = np.stack([1.0 / X[:, 1], -X[:, 0] / (X[:, 1] ** 2)], axis=1)
    reset_gs_reporter({"case": "ratio"})
    cfg = GeneralizedSymmetryConfig(enabled=True, mode="auto", min_confidence=0.5)
    props, diag = stageA_generalized_symmetry_proposals(
        atom=None, leaf=RatioLeaf(), x_vals=X, dydx_vals=G, cols=(0, 1), cfg=cfg
    )
    assert props
    assert any(d["accepted"] for d in diag)
    payload = write_gs_reports(
        json_path=tmp_path / "case.gs_report.json",
        markdown_path=tmp_path / "case.gs_report.md",
        final_expression="x0/x1",
        mode="auto",
    )
    assert payload["summary"]["satisfied_generators"] >= 1
    assert payload["summary"]["switched_off_generators"] >= 1
    loaded = json.loads((tmp_path / "case.gs_report.json").read_text())
    assert loaded["schema"] == "nestynet_sr_gs_report_v3"
    assert "x0/x1" in (tmp_path / "case.gs_report.md").read_text()
