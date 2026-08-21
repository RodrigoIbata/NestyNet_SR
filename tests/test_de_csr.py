# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from nestynet_sr.sr_de.de_csr import refine_factorized_search_candidate_from_runs


def test_refine_factorized_search_candidate_from_runs_accepts_improved_refinement(monkeypatch, tmp_path):
    x = np.linspace(0.0, 2.0 * np.pi, 240)
    u = -0.5 * np.cos(2.0 * x)
    csv_path = tmp_path / "traj.csv"
    csv_path.write_text(
        "y,x0\n" + "\n".join(f"{ui:.17g},{xi:.17g}" for ui, xi in zip(u, x)),
        encoding="utf-8",
    )
    run = SimpleNamespace(csv_path=csv_path, traj_id="ic0")
    candidate = {
        "engine": "factorized_search",
        "kind": "factorized",
        "order": 1,
        "x_axis": 0,
        "include_x": True,
        "include_u": False,
        "include_du": False,
        "feature_names": ["x0"],
        "expr_ast": ["sin", ["var", 0]],
        "mapping": {"kind": "poly", "coeffs": [0.0, 1.0]},
        "mapping_kind": "poly",
        "probe_mse": 1.0,
        "candidate_rank": 0,
    }

    def _fake_score_expr(node, *args, refine_enable=False, refine_cfg=None, refine_state=None, **kwargs):
        if refine_enable:
            refine_state["trials_done"] = int(refine_state.get("trials_done", 0)) + 1
            return (
                0.1,
                "refined",
                None,
                {"kind": "poly", "coeffs": [0.0, 1.0]},
                ("sin", ("mul", ("const", 2.0), ("var", 0))),
            )
        return (
            1.0,
            "base",
            None,
            {"kind": "poly", "coeffs": [0.0, 1.0]},
            ("sin", ("var", 0)),
        )

    monkeypatch.setattr("nestynet_sr.sr_de.de_csr.score_expr", _fake_score_expr)

    out = refine_factorized_search_candidate_from_runs(
        candidate,
        fit_runs=[run],
        probe_runs=[run],
        max_trials=4,
        seed=123,
    )

    assert out["accepted"] is True
    assert out["refined_probe_mse"] < out["base_probe_mse"]
    assert out["candidate"]["de_coe_csr_refined"] is True
    assert out["candidate"]["diagnostics"]["de_coe_csr"]["trials_used"] <= 4
