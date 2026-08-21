# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

import torch

from nestynet_sr.sr_search.factorized_search.emergent_basis import propose_emergent_basis_rows


TARGET_DIM = (0.0, -3.0, 1.0, 0.0, 0.0)
DIMLESS = (0.0, 0.0, 0.0, 0.0, 0.0)


def _pb037_data(n=900):
    g = torch.Generator().manual_seed(37)
    x0 = 1.0 + 4.0 * torch.rand((n, 1), generator=g, dtype=torch.float64)
    x1 = 1.0 + 4.0 * torch.rand((n, 1), generator=g, dtype=torch.float64)
    x2 = 1.0 + 4.0 * torch.rand((n, 1), generator=g, dtype=torch.float64)
    x = torch.cat([x0, x1, x2], dim=1)
    y = x0 + x1 + 2.0 * torch.sqrt(x0 * x1) * torch.cos(x2)
    return x[: n // 2], y[: n // 2], x[n // 2 :], y[n // 2 :]


def test_emergent_basis_promotes_target_dim_additive_subtree():
    x_fit, y_fit, x_probe, y_probe = _pb037_data()
    near_miss = (
        "mul",
        ("add", ("var", 0), ("var", 1)),
        ("cos", ("var", 2)),
    )
    stats = {}

    rows = propose_emergent_basis_rows(
        candidate_rows=[
            {
                "expr": near_miss,
                "proposal_key": "near_miss",
                "scaffold_family": "periodic",
                "scaffold_id": "periodic:0",
                "local_fit_mse": 1.0,
                "local_probe_mse": 1.0,
            }
        ],
        current_basis_state=None,
        x_fit=x_fit,
        y_fit=y_fit,
        x_probe=x_probe,
        y_probe=y_probe,
        var_dims=(TARGET_DIM, TARGET_DIM, DIMLESS),
        y_dims=TARGET_DIM,
        stats=stats,
        debug_limit=8,
    )

    assert len(rows) == 1
    assert rows[0]["emergent_basis_expr"] == "(x0+x1)"
    assert rows[0]["feature_block_obj"].family == "emergent_subexpr"
    assert rows[0]["local_probe_mse"] < rows[0]["emergent_basis_evidence"]["current_probe"]
    assert stats["emergent_basis_rows"] == 1
    assert stats["debug_emergent_basis"][0]["decision"] == "promote_row"


def test_emergent_basis_canonicalizes_commuted_additive_subtrees():
    x_fit, y_fit, x_probe, y_probe = _pb037_data()
    near_miss_a = (
        "mul",
        ("add", ("var", 0), ("var", 1)),
        ("cos", ("var", 2)),
    )
    near_miss_b = (
        "mul",
        ("add", ("var", 1), ("var", 0)),
        ("cos", ("var", 2)),
    )

    rows = propose_emergent_basis_rows(
        candidate_rows=[
            {
                "expr": near_miss_a,
                "proposal_key": "near_miss_a",
                "scaffold_family": "periodic",
                "scaffold_id": "periodic:0",
                "local_fit_mse": 1.0,
                "local_probe_mse": 1.0,
            },
            {
                "expr": near_miss_b,
                "proposal_key": "near_miss_b",
                "scaffold_family": "power",
                "scaffold_id": "power:0",
                "local_fit_mse": 1.1,
                "local_probe_mse": 1.1,
            },
        ],
        current_basis_state=None,
        x_fit=x_fit,
        y_fit=y_fit,
        x_probe=x_probe,
        y_probe=y_probe,
        var_dims=(TARGET_DIM, TARGET_DIM, DIMLESS),
        y_dims=TARGET_DIM,
        stats={},
        min_source_count=2,
    )

    assert len(rows) == 1
    evidence = rows[0]["emergent_basis_evidence"]
    assert rows[0]["emergent_basis_expr"] == "(x0+x1)"
    assert evidence["source_count"] == 2


def test_emergent_basis_rejects_wrong_dimension_subtree():
    x_fit, y_fit, x_probe, y_probe = _pb037_data()
    near_miss = (
        "mul",
        ("add", ("var", 0), ("var", 1)),
        ("cos", ("var", 2)),
    )
    stats = {}

    rows = propose_emergent_basis_rows(
        candidate_rows=[
            {
                "expr": near_miss,
                "proposal_key": "near_miss",
                "scaffold_family": "periodic",
                "scaffold_id": "periodic:0",
                "local_fit_mse": 1.0,
                "local_probe_mse": 1.0,
            }
        ],
        current_basis_state=None,
        x_fit=x_fit,
        y_fit=y_fit,
        x_probe=x_probe,
        y_probe=y_probe,
        var_dims=(TARGET_DIM, TARGET_DIM, DIMLESS),
        y_dims=DIMLESS,
        stats=stats,
    )

    assert rows == []
    assert stats["emergent_basis_reject_counts"]["dim_mismatch"] >= 1
