# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

from __future__ import annotations

import torch

from nestynet_sr.sr_search.factorized_search.subproblem_teacher import (
    build_numeric_local_teacher_spec,
    evaluate_local_teacher_jets,
)


class _QuadraticTeacher:
    def grad(self, x):
        return 2.0 * x

    def grad_grad(self, x):
        out = torch.zeros((int(x.shape[0]), int(x.shape[1]), int(x.shape[1])), dtype=x.dtype, device=x.device)
        out[:, 0, 0] = 2.0
        return out


def test_evaluate_local_teacher_jets_defaults_to_numeric_provenance():
    x = torch.linspace(-1.0, 1.0, 17, dtype=torch.float64).unsqueeze(-1)
    target = x * x

    jets = evaluate_local_teacher_jets(
        x,
        target,
        include_d2=True,
        max_rows=12,
    )

    assert jets["source"] == "numeric_local_quadratic"
    assert jets["requested_source"] == "numeric_local_quadratic"
    assert jets["fallback_used"] is False
    assert jets["grad"] is not None
    assert jets["d2"] is not None


def test_evaluate_local_teacher_jets_uses_runtime_teacher_when_available():
    x = torch.linspace(-1.0, 1.0, 9, dtype=torch.float64).unsqueeze(-1)
    target = x * x

    jets = evaluate_local_teacher_jets(
        x,
        target,
        include_d2=True,
        max_rows=8,
        teacher_spec={"source": "oracle"},
        teacher_runtime=_QuadraticTeacher(),
    )

    assert jets["source"] == "oracle"
    assert jets["requested_source"] == "oracle"
    assert jets["fallback_used"] is False
    assert torch.allclose(jets["grad"], 2.0 * x)
    assert jets["d2"] is not None
    assert torch.allclose(jets["d2"], torch.full_like(x, 2.0))


def test_build_numeric_local_teacher_spec_records_requested_source():
    spec = build_numeric_local_teacher_spec(requested_source="oracle", reason="unit_test")

    assert spec["source"] == "numeric_local_quadratic"
    assert spec["requested_source"] == "oracle"
    assert spec["reason"] == "unit_test"
