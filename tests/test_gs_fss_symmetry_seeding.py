# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata

import numpy as np


def _seed_nodes():
    from nestynet_sr.sr_core.bridges import Add, ConstNode, Exp, Mul, Pow, U, Var

    row = Mul(Mul(ConstNode(value=-1.0), Pow(Add(ConstNode(value=1.0), Var(0)), -1.0)), U())
    law = Mul(ConstNode(value=-0.9987), Mul(U(), Pow(Add(ConstNode(value=1.0), Var(0)), -1.0)))
    exp_row = Mul(ConstNode(value=-1.0), Exp(U()))
    return row, law, exp_row


def test_symmetry_seed_rows_compile_and_pin():
    from nestynet_sr.sr_search.factorized_search.oracle_lab_de import (
        DELabSpec,
        _gs_symmetry_seed_rows,
    )

    row, law, exp_row = _seed_nodes()
    spec = DELabSpec(
        id="t", csv_paths=(),
        extra={"gs_symmetry_seed_asts": (
            {"node": row, "label": "gs_row:u_over_1px"},
            {"node": law, "label": "gs_law:900"},
            {"node": exp_row, "label": "gs_row:exp_u"},
        )},
    )
    rows = _gs_symmetry_seed_rows(spec, 1)
    assert len(rows) == 3
    # strict priority over periodogram-hinted trig atoms (score -1.0) so an
    # aperiodic signal's spurious periodic hints cannot evict the seed
    assert all(r["score"] == -2.0 for r in rows)
    assert all(r["mapping_kind"] == "gs_symmetry_seed" for r in rows)
    exprs = " | ".join(r["expr"] for r in rows)
    assert "1+x0" in exprs and "exp(x1)" in exprs  # u maps to feature var x1


def test_symmetry_seed_rows_respect_feature_layout_and_gating():
    from nestynet_sr.sr_core.bridges import DU, Mul, Pow, Var
    from nestynet_sr.sr_search.factorized_search.oracle_lab_de import (
        DELabSpec,
        _gs_symmetry_seed_rows,
    )

    du_row = Mul(DU(0), Pow(Var(0), -1.0))  # u_x / x
    spec = DELabSpec(
        id="t2", csv_paths=(),
        extra={"gs_symmetry_seed_asts": ({"node": du_row, "label": "gs_row:ux_over_x"},)},
    )
    # du is a feature column only at order 2
    assert _gs_symmetry_seed_rows(spec, 1) == []
    rows2 = _gs_symmetry_seed_rows(spec, 2)
    assert len(rows2) == 1 and "x2" in rows2[0]["expr"]
    # no payload -> strict no-op
    assert _gs_symmetry_seed_rows(DELabSpec(id="e", csv_paths=()), 1) == []


def test_seed_lets_composer_assemble_exact_law():
    """A pinned symmetry seed must let the additive composer fit the exact RHS.

    Regression guard for the carrier-pool priority: the seed's score must beat
    the periodogram-hinted trig atoms so an aperiodic signal cannot evict it.
    """
    import torch

    from nestynet_sr.sr_core.bridges import Add, ConstNode, Mul, Pow, U, Var
    from nestynet_sr.sr_search.factorized_search.oracle_lab_de import (
        DELabSpec,
        _build_sparse_combo_rows,
        _gs_symmetry_seed_rows,
        default_oracle_de_hyperparams,
    )

    n = 400
    x = np.linspace(0.0, 3.0, n)
    u = 1.3 / (1.0 + x)
    ux = -1.3 / (1.0 + x) ** 2
    x_fit = torch.tensor(np.column_stack([x, u]), dtype=torch.float64)
    y_fit = torch.tensor(ux, dtype=torch.float64).reshape(-1, 1)

    row = Mul(Mul(ConstNode(value=-1.0), Pow(Add(ConstNode(value=1.0), Var(0)), -1.0)), U())
    spec = DELabSpec(
        id="t", csv_paths=(), include_du=False,
        extra={"gs_symmetry_seed_asts": ({"node": row, "label": "gs"},)},
    )
    seed_rows = _gs_symmetry_seed_rows(spec, 1)
    distractors = [
        {"expr": "x0", "_expr_obj": ("var", 0), "score": 0.5, "mapping_kind": "atom"},
        {"expr": "u", "_expr_obj": ("var", 1), "score": 0.4, "mapping_kind": "atom"},
    ]
    hp = default_oracle_de_hyperparams()
    hp.de_sparse_combo_enable = True
    hp.de_sparse_combo_max_terms = 3
    hp.de_sparse_combo_pool_topk = 6

    combo_rows = _build_sparse_combo_rows(
        spec=spec, order=1, base_rows=seed_rows + distractors,
        x_fit=x_fit, y_fit=y_fit, x_probe=x_fit, y_probe=y_fit,
        hp=hp, probe_meta=None, traj_metric="mean",
    )
    assert combo_rows, "composer produced no combo rows"
    best = min(combo_rows, key=lambda r: float(r.get("mse", 1e30)))
    assert float(best["mse"]) < 1e-10, f"expected exact fit, got mse={best.get('mse')}"
    assert any("1+x0" in s for s in best.get("combo_source_exprs", []))


def test_de_lab_spec_threads_seed_asts_from_cfg():
    from nestynet_sr.sr_de.de_search import DESearchConfig
    from nestynet_sr.sr_de.factorized_de import de_lab_spec_from_de_cfg

    row, law, _ = _seed_nodes()
    cfg = DESearchConfig(x_axis=0)
    spec_plain = de_lab_spec_from_de_cfg(cfg)
    assert not (spec_plain.extra or {}).get("gs_symmetry_seed_asts")

    cfg.gs_de_reduction_seed_asts = [
        {"node": row, "label": "gs_row:u_over_1px"},
        {"node": law, "label": "gs_law:900"},
    ]
    spec = de_lab_spec_from_de_cfg(cfg)
    payload = (spec.extra or {}).get("gs_symmetry_seed_asts")
    assert payload and len(payload) == 2


def test_run_de_hook_stashes_seed_asts(tmp_path):
    from types import SimpleNamespace

    from nestynet_sr.run_de import _attach_gs_reduction_rows

    x = np.linspace(0.0, 3.0, 500)
    paths = []
    for i, c in enumerate((0.7, 1.3, 2.1)):
        p = tmp_path / f"t{i}.csv"
        np.savetxt(p, np.column_stack([x, c / (1.0 + x)]), delimiter=",", header="x,y", comments="")
        paths.append(str(p))
    cfg = SimpleNamespace(order_candidates=[1], x_axis=0)
    _attach_gs_reduction_rows(cfg, paths, x_axis=0)
    seeds = getattr(cfg, "gs_de_reduction_seed_asts", [])
    assert seeds, "expected stashed seed ASTs"
    labels = {s["label"] for s in seeds}
    assert any(label.startswith("gs_row:") for label in labels)
    assert any(label.startswith("gs_law:") for label in labels)
