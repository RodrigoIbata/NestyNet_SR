# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata

from nestynet_sr.sr_core.bridges import AcosNode, AsinNode
from nestynet_sr.sr_search.factorized_search import bridge as bridge_mod
from nestynet_sr.sr_search.factorized_search.explorer import Rec


def test_factorized_search_bridge_converts_inverse_trig_tuple_nodes():
    asin_node = bridge_mod.factorized_search_to_nestynet(("asin", ("var", 0)))
    acos_node = bridge_mod.factorized_search_to_nestynet(("acos", ("var", 0)))

    assert isinstance(asin_node, AsinNode)
    assert isinstance(acos_node, AcosNode)


def test_run_explorer_exposes_raw_and_effective_mse(monkeypatch):
    class _FakeArch:
        def best(self, k, strategy="mse"):
            _ = (k, strategy)
            return [
                Rec(
                    best_mse=1.0e-2,      # effective/objective
                    best_expr=("var", 0),
                    visits=1,
                    mapping={"kind": "poly", "coeffs": [0.0, 1.0], "mu": 0.0, "std": 1.0},
                    z=None,
                    elites=[],
                    best_raw_mse=1.0e-4,  # raw
                )
            ]

    monkeypatch.setattr(bridge_mod, "run_explorer_core", lambda *args, **kwargs: _FakeArch())

    out = bridge_mod.run_explorer(
        target_fn=lambda x: x[:, :1],
        nvars=1,
        n_iter=1,
        max_depth=1,
        simplify_skeletons=False,
        return_topk=1,
    )
    assert len(out) == 1
    assert float(out[0]["mse"]) == 1.0e-4
    assert float(out[0]["mse_raw"]) == 1.0e-4
    assert float(out[0]["mse_eff"]) == 1.0e-2
